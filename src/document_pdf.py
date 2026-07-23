"""Render a Document's markdown content to a real, from-scratch PDF.

This is distinct from src/pdf_form_doc.py's export path, which only fills an
*existing* uploaded PDF template's form fields. There was previously no way to
turn an ordinary text/markdown Document into a downloadable PDF at all — the
agent's only option was to hand-roll a Python script (e.g. reportlab), which
routinely stalled: the file it wrote had no path back into the UI as a
download. This gives both the agent and the user a real, built-in export.

Uses fitz.Story (PyMuPDF, already a dependency for PDF form-filling) to lay
out sanitized HTML onto real PDF pages — no extra dependency needed.
"""

import logging
import tempfile

import markdown
import nh3

logger = logging.getLogger(__name__)

# Same allowlist shape as visual_report.py's report renderer — markdown source
# may be LLM-authored, so strip script/event-handler markup before it reaches
# fitz.Story's HTML layout engine.
_ALLOWED_TAGS = set(nh3.ALLOWED_TAGS) | {"details", "summary"}
_ALLOWED_ATTRS = {k: set(v) for k, v in nh3.ALLOWED_ATTRIBUTES.items()}

_PAGE_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 12pt 0; }
h2 { font-size: 15pt; margin: 18pt 0 8pt 0; }
h3 { font-size: 12.5pt; margin: 14pt 0 6pt 0; }
p { margin: 0 0 8pt 0; }
ul, ol { margin: 0 0 8pt 0; padding-left: 18pt; }
li { margin: 0 0 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 0 0 10pt 0; }
th, td { border: 0.5pt solid #999; padding: 4pt 6pt; text-align: left; font-size: 9.5pt; }
th { background-color: #eee; }
code { font-family: Courier, monospace; font-size: 9pt; background-color: #f2f2f2; padding: 1pt 3pt; }
pre { font-family: Courier, monospace; font-size: 9pt; background-color: #f2f2f2; padding: 6pt; white-space: pre-wrap; }
blockquote { border-left: 2pt solid #ccc; margin: 0 0 8pt 0; padding-left: 10pt; color: #444; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 12pt 0; }
"""


def _md_to_safe_html(md_text: str) -> str:
    html = markdown.markdown(
        md_text or "",
        extensions=["extra", "sane_lists"],
    )
    return nh3.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, link_rel=None)


def render_markdown_to_pdf(markdown_text: str, title: str = "") -> str:
    """Render markdown text to a PDF file and return its path (caller unlinks it).

    Raises RuntimeError on failure (e.g. PyMuPDF not installed) — callers
    should surface this as a clear error rather than a silent no-op.
    """
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError(
            "PDF export requires PyMuPDF, which is not installed on this server."
        ) from e

    body_html = _md_to_safe_html(markdown_text)
    heading_html = f"<h1>{nh3.clean(title)}</h1>" if title and title.strip() else ""
    full_html = f"<html><body>{heading_html}{body_html}</body></html>"

    out_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    try:
        story = fitz.Story(html=full_html, user_css=_PAGE_CSS)
        writer = fitz.DocumentWriter(out_path)
        mediabox = fitz.paper_rect("a4")
        where = mediabox + (44, 44, -44, -44)
        more = True
        while more:
            device = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(device)
            writer.end_page()
        writer.close()
    except Exception as e:
        logger.error("Markdown-to-PDF rendering failed: %s", e)
        raise RuntimeError(f"PDF rendering failed: {e}") from e

    return out_path
