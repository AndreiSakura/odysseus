"""Pins the fixes from a transcription chat that burned ~640k tokens:

- create_document can load its body from <source_path> instead of the model
  re-typing a whole transcript it already has on disk.
- bash background detection accepts a trailing `#!bg` and a bare trailing `&`
  (a bare `&` used to detach outside the job system, so the model re-ran the
  whole transcription in the foreground).
"""

import asyncio

import pytest

from src.agent_tools.document_tools import CreateDocumentTool
from src.tool_execution import _split_bg_marker


@pytest.mark.parametrize("cmd,expected", [
    ("#!bg\npython3 x.py", (True, "python3 x.py")),
    ("python3 x.py --out a.md #!bg", (True, "python3 x.py --out a.md")),
    ("python3 x.py --out a.md &", (True, "python3 x.py --out a.md")),
    ("python3 x.py > log 2>&1 &", (True, "python3 x.py > log 2>&1")),
    ("make && make install", (False, "make && make install")),
    ("echo hi", (False, "echo hi")),
    ("sleep 1 &\nwait", (False, "sleep 1 &\nwait")),
])
def test_split_bg_marker_forms(cmd, expected):
    assert _split_bg_marker(cmd) == expected


class _FakeQuery:
    def filter(self, *a, **k):
        return self

    def first(self):
        return None


class _Row:
    id = None  # queried as a column (DbSession.id == session_id)

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeDB:
    def __init__(self):
        self.added = []

    def query(self, *a, **k):
        return _FakeQuery()

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def test_create_document_loads_source_path(tmp_path, monkeypatch):
    src_file = tmp_path / "call_transcription.md"
    src_file.write_text("**Speaker 1** [00:00:01]\nHello\n", encoding="utf-8")

    import src.database as database
    import src.tool_execution as tool_execution
    db = _FakeDB()
    monkeypatch.setattr(database, "SessionLocal", lambda: db, raising=False)
    for name in ("Document", "DocumentVersion", "Session"):
        monkeypatch.setattr(database, name, _Row, raising=False)
    monkeypatch.setattr(tool_execution, "_resolve_tool_path", lambda p: p)

    raw = f"call_transcription\nmarkdown\n<source_path>{src_file}</source_path>\n"
    result = asyncio.run(CreateDocumentTool().execute(raw, {"session_id": "s1"}))

    assert result["action"] == "create"
    assert result["title"] == "call_transcription"
    assert result["content"] == "**Speaker 1** [00:00:01]\nHello\n"
    assert db.added[0].current_content == result["content"]


def test_create_document_source_path_outside_roots_errors(monkeypatch):
    import src.database as database
    import src.tool_execution as tool_execution
    for name in ("SessionLocal", "Document", "DocumentVersion", "Session"):
        monkeypatch.setattr(database, name, _Row, raising=False)

    def _reject(p):
        raise ValueError(f"path '{p}' is outside the allowed roots")

    monkeypatch.setattr(tool_execution, "_resolve_tool_path", _reject)
    raw = "t\n<source_path>/etc/passwd</source_path>"
    result = asyncio.run(CreateDocumentTool().execute(raw, {"session_id": "s1"}))
    assert "error" in result and "outside the allowed roots" in result["error"]
