"""A scheduled Skills Audit ran while the model endpoint was down and:

- got an "inconclusive" verdict, then demoted a working published skill to
  draft (hiding it from the agent's skill index);
- re-saved it via update_skill, which re-emitted the body from parsed
  sections, flattening a multi-line code block into one line and dropping the
  "## Notes" heading.
"""

import textwrap

import routes.skills_routes as skills_routes
from services.memory.skills import SkillsManager

_SKILL = textwrap.dedent("""\
    ---
    name: demo-skill
    description: Demo
    version: 1.0.0
    category: general
    status: published
    confidence: 0.95
    source: curated
    owner: admin
    created: "2026-10-01T10:45:32Z"
    ---

    ## When to Use

    Whenever.

    ## Procedure

    1. **Run it in the background.** First line must be the marker:
       ```
       #!bg
       python3 run.py --input <path>
       ```
       Do not use `&`.

    2. Second step.

    ## Notes

    - A note that must keep its heading.
    """)


def _manager(tmp_path):
    path = tmp_path / "skills" / "general" / "demo-skill" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(_SKILL, encoding="utf-8")
    return SkillsManager(str(tmp_path)), path


def test_metadata_update_keeps_body_verbatim(tmp_path):
    mgr, path = _manager(tmp_path)
    assert mgr.update_skill("demo-skill", {"status": "draft", "confidence": 0.5}, owner="admin")

    text = path.read_text(encoding="utf-8")
    assert "status: draft" in text
    assert "confidence: 0.5" in text
    body = _SKILL.split("---\n\n", 1)[1]
    assert text.endswith(body)


def test_inconclusive_audit_keeps_published_status(tmp_path, monkeypatch):
    mgr, path = _manager(tmp_path)
    monkeypatch.setattr(skills_routes, "_audit_auto_publish_policy", lambda owner: (True, 0.7))

    status = skills_routes._audit_finalize_status(mgr, "demo-skill", "admin", "inconclusive", 0.95)

    assert status == "published"
    assert "status: published" in path.read_text(encoding="utf-8")


def test_failing_audit_still_demotes(tmp_path, monkeypatch):
    mgr, path = _manager(tmp_path)
    monkeypatch.setattr(skills_routes, "_audit_auto_publish_policy", lambda owner: (True, 0.7))

    status = skills_routes._audit_finalize_status(mgr, "demo-skill", "admin", "fail", 0.4)

    assert status == "draft"
    assert "status: draft" in path.read_text(encoding="utf-8")
