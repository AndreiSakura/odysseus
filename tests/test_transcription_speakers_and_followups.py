"""Transcription workflow fixes from the 6–7 Oct sessions:

- scripts/speakers.py previews and renames diarized speakers in one exact pass
  (the agent used to hand-edit labels for dozens of rounds);
- update_document can target a document by id and load a file;
- a ```bash fence marked `#!bg` runs even for native tool-calling models (it
  was silently skipped while the reply said the job had started);
- a user turn waits for a running background follow-up instead of running a
  second agent in parallel;
- the history API reports a pending background job so the page keeps watching.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("speakers", ROOT / "scripts" / "speakers.py")
speakers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speakers)

TRANSCRIPT = (
    "**Speaker 1** [00:00:05]\nHey Andrei, can you hear me?\n\n"
    "**Speaker 2** [00:00:09]\nYes, loud and clear. Let's go through the open questions.\n\n"
    "**Speaker 1** [00:01:10]\nOkay, question six is about the engineering requests.\n\n"
    "**Speaker 3** [00:02:00]\nI'll take the action on that one.\n"
)


def test_preview_lists_each_speaker_with_samples():
    out = speakers.preview(TRANSCRIPT)
    assert "SPEAKERS (3 detected" in out
    assert "- Speaker 1: 2 turns" in out
    assert '[00:00:09] "Yes, loud and clear.' in out
    assert "talking TO" in out


def test_rename_by_label_and_timestamp():
    text, summary = speakers.rename(TRANSCRIPT, ["Speaker 1=Don", "Speaker 2=Andrei", "00:02:00=Denise"])
    assert "**Don** [00:00:05]" in text and "**Don** [00:01:10]" in text
    assert "**Andrei** [00:00:09]" in text
    assert "**Denise** [00:02:00]" in text
    assert "Speaker 1 -> Don: 2 turn(s)" in summary
    # Body text is never touched, only the turn headers.
    assert "Hey Andrei, can you hear me?" in text


def test_rename_swaps_labels_simultaneously():
    text, _ = speakers.rename(TRANSCRIPT, ["Speaker 1=Speaker 2", "Speaker 2=Speaker 1"])
    assert "**Speaker 2** [00:00:05]" in text
    assert "**Speaker 1** [00:00:09]" in text


def test_rename_rejects_unknown_label_or_timestamp():
    with pytest.raises(ValueError, match="Speaker 9"):
        speakers.rename(TRANSCRIPT, ["Speaker 9=X"])
    with pytest.raises(ValueError, match="00:09:99"):
        speakers.rename(TRANSCRIPT, ["00:09:99=X"])


def test_rename_cli_rewrites_file_in_place(tmp_path, capsys):
    p = tmp_path / "call_transcription.md"
    p.write_text(TRANSCRIPT, encoding="utf-8")
    assert speakers.main(["rename", str(p), "Speaker 3=Denise"]) == 0
    assert "**Denise** [00:02:00]" in p.read_text(encoding="utf-8")
    # A bad mapping leaves the file untouched.
    before = p.read_text(encoding="utf-8")
    assert speakers.main(["rename", str(p), "Speaker 7=X"]) == 1
    assert p.read_text(encoding="utf-8") == before


# ---- update_document: explicit doc id + source_path ------------------------

class _Doc:
    def __init__(self, doc_id, title, content):
        self.id, self.title, self.current_content = doc_id, title, content
        self.language, self.version_count = "markdown", 1


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _DB:
    def __init__(self):
        self.added = []

    def add(self, o):
        self.added.append(o)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _patch_update(monkeypatch, docs):
    import src.database as database
    import src.tool_execution as tool_execution
    from src.agent_tools import document_tools

    db = _DB()
    monkeypatch.setattr(database, "SessionLocal", lambda: db, raising=False)
    monkeypatch.setattr(database, "Document", _Row, raising=False)
    monkeypatch.setattr(database, "DocumentVersion", _Row, raising=False)
    monkeypatch.setattr(tool_execution, "_resolve_tool_path", lambda p: p)
    monkeypatch.setattr(document_tools, "_get_owned_document",
                        lambda db_, D, doc_id, owner, active_only=False: docs.get(doc_id))
    monkeypatch.setattr(document_tools, "_most_recent_owned_document",
                        lambda db_, D, owner, active_only=False: docs.get("summary"))
    return db


def test_update_document_targets_doc_id_and_loads_file(tmp_path, monkeypatch):
    from src.agent_tools.document_tools import UpdateDocumentTool

    docs = {"tx": _Doc("tx", "call_transcription", "**Speaker 1** [00:00:05]\nhi\n"),
            "summary": _Doc("summary", "call_summary", "# Summary")}
    _patch_update(monkeypatch, docs)
    f = tmp_path / "call_transcription.md"
    f.write_text("**Don** [00:00:05]\nhi\n", encoding="utf-8")

    raw = f"<doc_id>tx</doc_id>\n<source_path>{f}</source_path>\n"
    result = asyncio.run(UpdateDocumentTool().execute(raw, {"session_id": "s1"}))

    assert result["doc_id"] == "tx" and result["version"] == 2
    assert docs["tx"].current_content == "**Don** [00:00:05]\nhi"
    assert docs["summary"].current_content == "# Summary"  # the wrong doc is left alone


def test_update_document_unknown_doc_id_does_not_fall_back(monkeypatch):
    from src.agent_tools.document_tools import UpdateDocumentTool

    docs = {"summary": _Doc("summary", "call_summary", "# Summary")}
    _patch_update(monkeypatch, docs)
    result = asyncio.run(UpdateDocumentTool().execute("<doc_id>nope</doc_id>\nnew text", {}))
    assert "not found" in result["error"]
    assert docs["summary"].current_content == "# Summary"


def test_update_document_schema_args_become_tags():
    from src.tool_schemas import function_call_to_tool_block

    block = function_call_to_tool_block(
        "update_document", '{"doc_id": "tx", "source_path": "/app/data/x.md"}')
    assert "<doc_id>tx</doc_id>" in block.content
    assert "<source_path>/app/data/x.md</source_path>" in block.content


# ---- #!bg fences for native models -----------------------------------------

def test_bg_marked_fence_runs_for_native_models():
    from src.agent_loop import _bg_marked_fenced_bash

    reply = ("Starting it now:\n\n```bash\n#!bg python3 /app/scripts/transcribe_media.py "
             "--input /app/data/a.mp4 --output /app/data/a_transcription.md\n```\n")
    blocks = _bg_marked_fenced_bash(reply)
    assert len(blocks) == 1 and blocks[0].tool_type == "bash"
    assert "transcribe_media.py" in blocks[0].content


@pytest.mark.parametrize("reply", [
    "Here is how you'd do it:\n```bash\npython3 x.py\n```",          # illustrative, unmarked
    "Checking:\n```bash\n#!bg manage_bg_jobs action=list\n```",       # a tool name, not a program
])
def test_unmarked_or_tool_name_fences_are_ignored(reply):
    from src.agent_loop import _bg_marked_fenced_bash

    assert _bg_marked_fenced_bash(reply) == []


# ---- follow-up concurrency and pending flag --------------------------------

def test_user_turn_waits_for_running_followup():
    from src import agent_runs

    order = []

    async def followup():
        await asyncio.sleep(0.05)
        order.append("followup saved")
        yield "data: [DONE]\n\n"

    async def scenario():
        agent_runs.start("sess-wait", followup(), kind="bg_followup")
        assert agent_runs.followup_running("sess-wait")
        waited = await agent_runs.wait_for_followup("sess-wait")
        order.append("user turn")
        return waited

    assert asyncio.run(scenario()) is True
    assert order == ["followup saved", "user turn"]


def test_no_wait_for_ordinary_chat_runs():
    from src import agent_runs

    async def chat():
        await asyncio.sleep(1)
        yield "data: [DONE]\n\n"

    async def scenario():
        agent_runs.start("sess-chat", chat())
        try:
            return await agent_runs.wait_for_followup("sess-chat")
        finally:
            agent_runs.stop("sess-chat")

    assert asyncio.run(scenario()) is False


@pytest.mark.parametrize("jobs,expected", [
    ([{"session_id": "s", "status": "running"}], True),
    ([{"session_id": "s", "status": "done", "followed_up": False}], True),
    ([{"session_id": "s", "status": "done", "followed_up": True}], False),
    ([{"session_id": "other", "status": "running"}], False),
])
def test_session_has_pending(monkeypatch, jobs, expected):
    from src import bg_jobs

    monkeypatch.setattr(bg_jobs, "refresh", lambda: {str(i): j for i, j in enumerate(jobs)})
    assert bg_jobs.session_has_pending("s") is expected


# ---- delivering files: tools always offered, hand-built .docx refused -----

def test_document_tools_always_offered():
    from src.tool_index import ALWAYS_AVAILABLE

    assert {"create_document", "update_document", "manage_documents"} <= ALWAYS_AVAILABLE


@pytest.mark.parametrize("code,blocked", [
    ("from docx import Document\nd = Document()\nd.add_paragraph('x')\nd.save('/app/data/a.docx')", True),
    ("python3 << 'EOF'\nimport docx\ndoc = docx.Document()\ndoc.save('t.docx')\nEOF", True),
    ("from reportlab.platypus import SimpleDocTemplate\nSimpleDocTemplate('a.pdf').build([])", True),
    # Reading an uploaded .docx (comparing transcripts) must keep working.
    ("from docx import Document\nd = Document('/app/data/uploads/x.docx')\nprint(len(d.paragraphs))", False),
    ("python3 /app/scripts/speakers.py rename a.md 'Speaker 1=Don'", False),
])
def test_builds_office_file(code, blocked):
    from src.tool_execution import _builds_office_file

    assert _builds_office_file(code) is blocked


def test_hand_built_docx_is_refused_with_guidance(monkeypatch):
    from src.tool_parsing import ToolBlock  # before tool_execution: avoids a circular import
    import src.tool_execution as tool_execution
    from src.tool_execution import _execute_tool_block_impl

    # Run as an admin so the earlier admin-only gate doesn't answer first.
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: True)

    code = "from docx import Document\nd = Document()\nd.save('/app/data/t.docx')"
    desc, result = asyncio.run(_execute_tool_block_impl(ToolBlock("python", code), session_id="s1"))
    assert "BLOCKED" in desc
    assert "create_document" in result["error"] and result["exit_code"] == 1
