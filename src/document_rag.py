"""Wire the Document library (DB rows) into the same Chroma RAG index used
for personal-doc folders, so a document that's "in the library" is actually
retrievable by relevance instead of requiring a full re-attachment to see it
again in a later chat turn.

Chunks are written into the same collection/lanes personal_docs.py populates
(src.rag_singleton.get_rag_manager() -> VectorRAG), tagged with
source=f"document:{doc_id}" so they can be found and removed as a unit later.
The existing RAG search path in chat_processor.build_context_preface already
queries this same collection, so no retrieval-side change is needed — indexing
a Document here is enough to make it show up in "retrieved documents" context.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Skip near-empty scratch docs (fresh "Untitled" tabs, one-line notes) so the
# index isn't churned with content too short to ever be a useful RAG hit.
MIN_INDEXABLE_CHARS = 200


def _source_for(doc_id: str) -> str:
    return f"document:{doc_id}"


def index_document_for_rag(doc_id: str, title: str, content: str, owner: Optional[str]) -> None:
    """Chunk and (re-)index a Document's content into the RAG store.

    Safe to call on every create/update: clears any chunks already indexed
    under this doc_id first, so an edit never leaves stale chunks alongside
    the fresh ones. No-ops quietly if RAG isn't available (ChromaDB down) or
    the content is too short to be worth indexing — this path must never
    raise into a document-save request.
    """
    text = (content or "").strip()
    try:
        from src.rag_singleton import get_rag_manager
        rag = get_rag_manager()
        if not rag:
            return

        rag.delete_by_source(_source_for(doc_id))
        if len(text) < MIN_INDEXABLE_CHARS:
            return

        from src.personal_docs import split_chunks
        chunks = split_chunks(text)
        meta_base = {
            "source": _source_for(doc_id),
            "filename": title or doc_id,
            "type": "document",
            "doc_id": doc_id,
        }
        if owner:
            meta_base["owner"] = owner

        indexed = 0
        for i, chunk in enumerate(chunks):
            if rag.add_document(chunk, {**meta_base, "chunk_id": i}):
                indexed += 1
        logger.info("Indexed %d/%d chunks for document %s into RAG", indexed, len(chunks), doc_id)
    except Exception:
        logger.warning("RAG indexing failed for document %s", doc_id, exc_info=True)


def remove_document_from_rag(doc_id: str) -> None:
    """Remove all RAG chunks previously indexed for this document."""
    try:
        from src.rag_singleton import get_rag_manager
        rag = get_rag_manager()
        if rag:
            rag.delete_by_source(_source_for(doc_id))
    except Exception:
        logger.warning("RAG removal failed for document %s", doc_id, exc_info=True)
