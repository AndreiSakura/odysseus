#!/usr/bin/env python3
"""Preview and rename speakers in a transcript written by transcribe_media.py.

Diarization can only tell voices apart ("Speaker 1", "Speaker 2", ...); it
never knows names. These two commands let the agent show the user who said
what, then apply the names the user confirms in one exact pass on the file,
instead of guessing or hand-editing a long document line by line.

Usage:
    python3 scripts/speakers.py preview TRANSCRIPT.md
    python3 scripts/speakers.py rename TRANSCRIPT.md "Speaker 1=Don" "Speaker 2=Andrei"
    python3 scripts/speakers.py rename TRANSCRIPT.md "00:05:03=Andrei" "00:06:15=Andrei"

rename mappings:
    "<label>=<name>"     every turn currently labelled <label> (e.g. "Speaker 3"
                         or a name applied earlier) becomes <name>.
    "<HH:MM:SS>=<name>"  only the turn starting at that timestamp becomes <name>
                         (per-line corrections). Applied after label mappings.
Label mappings are applied simultaneously, so "Speaker 1=Speaker 2" and
"Speaker 2=Speaker 1" swap rather than collapse. The file is rewritten in
place; a summary of what changed is printed. Exits non-zero, changing
nothing, if a label or timestamp doesn't exist in the transcript.
"""
import re
import sys
from collections import OrderedDict

TURN_RE = re.compile(r"^\*\*(?P<label>[^*\n]+)\*\* \[(?P<ts>\d{2}:\d{2}:\d{2})\]$")
TS_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


def parse_turns(text: str):
    """Return [(line_index, label, timestamp, first_text_line)] for each turn header."""
    lines = text.split("\n")
    turns = []
    for i, line in enumerate(lines):
        m = TURN_RE.match(line)
        if m:
            body = lines[i + 1].strip() if i + 1 < len(lines) else ""
            turns.append((i, m.group("label"), m.group("ts"), body))
    return turns


def preview(text: str, samples: int = 3, width: int = 140) -> str:
    """Per speaker: turn count, word count, and sample lines spread over the call."""
    turns = parse_turns(text)
    if not turns:
        return "No speaker-labelled turns found (diarization may have been skipped)."
    lines = text.split("\n")
    by_label = OrderedDict()
    for idx, label, ts, _ in turns:
        by_label.setdefault(label, []).append((idx, ts))

    def turn_text(idx):
        body = []
        for ln in lines[idx + 1:]:
            if TURN_RE.match(ln):
                break
            body.append(ln.strip())
        return " ".join(b for b in body if b)

    out = [f"SPEAKERS ({len(by_label)} detected — names below are labels, not identities):"]
    for label, items in by_label.items():
        words = sum(len(turn_text(i).split()) for i, _ in items)
        out.append(f"- {label}: {len(items)} turns, ~{words} words")
        # Prefer substantive turns; spread picks across the call.
        rich = [(i, ts) for i, ts in items if len(turn_text(i).split()) >= 6] or items
        step = max(1, len(rich) // samples)
        for i, ts in rich[::step][:samples]:
            t = turn_text(i)
            out.append(f'    [{ts}] "{t[:width]}{"…" if len(t) > width else ""}"')
    out.append(
        "A name a speaker SAYS is usually the person they are talking TO, not "
        "themselves. Confirm names with the user before relabelling."
    )
    return "\n".join(out)


def rename(text: str, mappings):
    """Apply mappings; return (new_text, summary_lines). Raises ValueError on bad input."""
    label_map, ts_map = {}, {}
    for m in mappings:
        if "=" not in m:
            raise ValueError(f"bad mapping {m!r}: expected '<label or HH:MM:SS>=<name>'")
        key, name = (p.strip() for p in m.split("=", 1))
        key = key.strip("*").strip()
        if not key or not name:
            raise ValueError(f"bad mapping {m!r}: empty side")
        if TS_RE.match(key):
            ts_map[key.zfill(8)] = name
        else:
            label_map[key] = name

    turns = parse_turns(text)
    labels = {label for _, label, _, _ in turns}
    stamps = {}
    for idx, label, ts, _ in turns:
        stamps.setdefault(ts, []).append(idx)
    missing = [k for k in label_map if k not in labels] + [k for k in ts_map if k not in stamps]
    if missing:
        raise ValueError(
            "not in transcript: " + ", ".join(missing)
            + f". Labels present: {', '.join(sorted(labels))}"
        )

    lines = text.split("\n")
    changed = {}
    for idx, label, ts, _ in turns:
        new = label_map.get(label, label)
        if ts in ts_map:
            new = ts_map[ts]
        if new != label:
            lines[idx] = f"**{new}** [{ts}]"
            changed[(label, new)] = changed.get((label, new), 0) + 1
    summary = [f"{old} -> {new}: {n} turn(s)" for (old, new), n in changed.items()]
    dup_ts = [ts for ts in ts_map if len(stamps[ts]) > 1]
    if dup_ts:
        summary.append("note: several turns start at " + ", ".join(dup_ts) + "; all of them were renamed")
    return "\n".join(lines), summary


def main(argv):
    if len(argv) < 2 or argv[0] not in ("preview", "rename"):
        print(__doc__, file=sys.stderr)
        return 2
    cmd, path, rest = argv[0], argv[1], argv[2:]
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if cmd == "preview":
        print(preview(text))
        return 0
    if not rest:
        print("ERROR: rename needs at least one mapping", file=sys.stderr)
        return 2
    try:
        new_text, summary = rename(text, rest)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    print("\n".join(summary) if summary else "No changes (names already applied).")
    print(preview(new_text))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
