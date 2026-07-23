#!/usr/bin/env python3
"""Transcribe an audio/video file to a speaker-labeled Markdown transcript.

Pipeline: ffmpeg (extract mono 16kHz audio) -> faster-whisper (speech to
text, word-level timestamps) -> pyannote.audio (speaker diarization) ->
word-level speaker assignment -> merged, labeled transcript.

Usage:
    python3 scripts/transcribe_media.py --input meeting.mp4 [options]

Options:
    --output PATH        Write transcript here instead of stdout.
    --model SIZE         faster-whisper model size (default: small).
                          One of tiny/base/small/medium/large-v3, or a
                          distil-whisper repo id.
    --language CODE      Force a language (e.g. "en"); default: auto-detect.
    --no-diarize         Skip speaker diarization, plain transcript only.
    --num-speakers N      Exact speaker count, if known.
    --min-speakers N      Lower bound on speaker count.
    --max-speakers N      Upper bound on speaker count.

Requires ffmpeg on PATH, `faster-whisper` installed, and for diarization
`pyannote.audio` plus an HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var whose
account has accepted the license for pyannote/speaker-diarization-3.1 and
pyannote/segmentation-3.0 on huggingface.co.

Prints progress lines to stderr; the transcript itself goes to stdout (or
--output). Exits non-zero with a plain-English message on any failure.
"""
import argparse
import os
import subprocess
import sys
import tempfile


def _fix_user_site_packages() -> None:
    """Some callers (e.g. Odysseus's agent bash tool) override $HOME for
    subprocesses, which silently redirects Python's user-site package
    resolution (faster-whisper/pyannote/torch live under the *real*
    HOME's ~/.local) to the wrong directory instead of raising an error —
    imports below would just fail as "not installed". Resolve the real
    home via the passwd database (unaffected by $HOME) and add its
    site-packages dir directly, ahead of whatever $HOME currently is."""
    try:
        import pwd
        import site
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        if real_home and real_home != os.environ.get("HOME"):
            candidate = os.path.join(
                real_home, ".local", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
            )
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
    except Exception:
        pass


_fix_user_site_packages()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def extract_audio(input_path: str, out_wav: str) -> None:
    ffmpeg = None
    for candidate in ("ffmpeg",):
        from shutil import which
        if which(candidate):
            ffmpeg = candidate
            break
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. Install it (e.g. `apt-get install ffmpeg`).")

    cmd = [ffmpeg, "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-vn", out_wav]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{proc.stderr[-2000:]}")


def pick_device():
    try:
        import torch
        return ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "int8")
    except Exception:
        return ("cpu", "int8")


def _add_nvidia_lib_dirs_to_ld_path() -> None:
    """Several CUDA-using libraries in this pipeline dlopen() their CUDA
    libs at runtime instead of via rpath, and don't all agree on which CUDA
    version: faster-whisper's CTranslate2 backend wants CUDA-12 cuBLAS/cuDNN
    (`nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip packages), while
    pyannote's torchcodec audio decoder wants torch's own CUDA-13 nvrtc
    (`nvidia-cu13`, already installed as a torch dependency). None of these
    .so directories are on the loader's default search path — add every
    `site-packages/nvidia/*/lib` dir to LD_LIBRARY_PATH before anything
    CUDA-related is imported."""
    try:
        import nvidia
        bases = list(nvidia.__path__)  # namespace package: no __file__, only __path__
    except Exception:
        return
    lib_dirs = []
    for base in bases:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            lib_dir = os.path.join(base, entry, "lib")
            if os.path.isdir(lib_dir):
                lib_dirs.append(lib_dir)
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))


def transcribe(wav_path: str, model_size: str, language: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper not installed. Install optional dependencies with "
            "`pip install -r requirements-optional.txt` (or `pip install faster-whisper`)."
        ) from exc

    device, compute_type = pick_device()

    def load_and_run():
        log(f"Loading faster-whisper model '{model_size}' on {device}...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log("Transcribing...")
        segments, info = model.transcribe(
            wav_path,
            language=language or None,
            word_timestamps=True,
            vad_filter=True,
        )
        return list(segments), info  # force generator to run now, inside the try/except

    try:
        segments, info = load_and_run()
    except Exception as exc:
        if device == "cuda":
            # faster-whisper's CTranslate2 backend needs its own CUDA runtime
            # libs (cuBLAS/cuDNN) separate from torch's bundled ones — torch
            # reporting CUDA available doesn't guarantee those are present,
            # and the failure only surfaces once transcription actually runs
            # (model construction is lazy).
            log(f"WARNING: GPU transcription failed ({exc}); falling back to CPU.")
            device, compute_type = "cpu", "int8"
            segments, info = load_and_run()
        else:
            raise

    words = []  # (start, end, text)
    for seg in segments:
        if not seg.words:
            # Segment without word timestamps (rare) — treat as one word-span.
            words.append((seg.start, seg.end, seg.text.strip()))
            continue
        for w in seg.words:
            text = w.word.strip()
            if text:
                words.append((w.start, w.end, text))

    log(f"Detected language: {info.language} (p={info.language_probability:.2f})")
    return words


def _load_waveform(wav_path: str):
    """Load audio into a {'waveform', 'sample_rate'} dict for pyannote,
    bypassing its default torchcodec-based file loader — torchcodec's
    prebuilt binaries are ABI-pinned to specific torch releases and fail
    against newer torch builds (e.g. 2.11.0+cu130) once the audio is more
    than a few seconds long, even though import-time only warns."""
    import soundfile as sf
    import torch
    data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (channel, time), matches pyannote's expectation
    return {"waveform": waveform, "sample_rate": sample_rate}


def diarize(wav_path: str, num_speakers=None, min_speakers=None, max_speakers=None):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF_TOKEN / HUGGING_FACE_HUB_TOKEN set. Speaker diarization needs a "
            "HuggingFace token from an account that has accepted the license for "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0."
        )
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio not installed. Install it with `pip install pyannote.audio`."
        ) from exc

    log("Loading pyannote speaker-diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)

    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    log("Running diarization (this can take a while for long files)...")
    audio_input = _load_waveform(wav_path)
    output = pipeline(audio_input, **kwargs)
    # pyannote.audio >=4 wraps the result in a DiarizeOutput dataclass; older
    # 3.x returns the Annotation directly. exclusive_speaker_diarization is
    # the non-overlapping-turns Annotation we want for word-level matching.
    diarization = getattr(output, "exclusive_speaker_diarization", output)

    turns = []  # (start, end, raw_label)
    for turn, _, label in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, label))
    turns.sort(key=lambda t: t[0])
    return turns


def speaker_for(word_start: float, word_end: float, turns) -> str:
    """Raw diarization label with the largest time-overlap with [word_start, word_end]."""
    best_label = None
    best_overlap = 0.0
    for t_start, t_end, label in turns:
        overlap = min(word_end, t_end) - max(word_start, t_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def assign_speakers(words, turns):
    """Map each word to a raw pyannote label, then renumber labels as
    "Speaker 1", "Speaker 2", ... in order of first appearance."""
    label_order = []
    label_map = {}
    tagged = []
    for start, end, text in words:
        raw = speaker_for(start, end, turns) if turns else None
        if raw is None:
            speaker = tagged[-1][2] if tagged else "Speaker 1"
        else:
            if raw not in label_map:
                label_order.append(raw)
                label_map[raw] = f"Speaker {len(label_order)}"
            speaker = label_map[raw]
        tagged.append((start, end, speaker, text))
    return tagged


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_transcript(tagged) -> str:
    if not tagged:
        return "*(no speech detected)*\n"

    lines = []
    cur_speaker = None
    cur_start = None
    cur_words = []

    def flush():
        if cur_words:
            text = " ".join(cur_words).strip()
            if cur_speaker:
                lines.append(f"**{cur_speaker}** [{format_timestamp(cur_start)}]\n{text}\n")
            else:
                lines.append(f"[{format_timestamp(cur_start)}]\n{text}\n")

    for start, end, speaker, text in tagged:
        if speaker != cur_speaker:
            flush()
            cur_speaker = speaker
            cur_start = start
            cur_words = [text]
        else:
            cur_words.append(text)
    flush()
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the audio/video file")
    parser.add_argument("--output", help="Write transcript here instead of stdout")
    parser.add_argument("--model", default="small", help="faster-whisper model size (default: small)")
    parser.add_argument("--language", default="", help="Force language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        log(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    _add_nvidia_lib_dirs_to_ld_path()

    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        log(f"Extracting audio from {args.input}...")
        extract_audio(args.input, tmp_wav)

        words = transcribe(tmp_wav, args.model, args.language)

        turns = []
        if not args.no_diarize:
            try:
                turns = diarize(
                    tmp_wav,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
            except Exception as exc:
                log(f"WARNING: diarization skipped ({exc}). Continuing with a plain transcript.")
                turns = []

        tagged = assign_speakers(words, turns)
        transcript = build_transcript(tagged)

        if args.output:
            with open(args.output, "w") as f:
                f.write(transcript)
            log(f"DONE {args.output}")
        else:
            print(transcript)
            log("DONE (stdout)")

    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    main()
