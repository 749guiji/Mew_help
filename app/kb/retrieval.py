"""Online retrieval: embed a query, search Milvus, fetch the answer from MySQL.

This is the read half of the ch03 knowledge-base pipeline. A query is embedded
with :class:`app.kb.embedding.Embedder`, the vector is searched against
:class:`app.kb.milvus_store.MilvusStore`, and the top hit's id is used to load
the source-of-truth row from ``knowledge_chunks`` (MySQL) so the returned answer
is always the authoritative text, not a copy cached in the vector store.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.db import SessionLocal
from app.db.models import KnowledgeChunk
from app.kb.embedding import Embedder
from app.kb.milvus_store import MilvusStore


def retrieve(
    session,
    store,
    embedder,
    query: str,
    *,
    top_k: int = 1,
    threshold: float = 0.5,
) -> str | None:
    """Return the answer of the best-matching chunk, or ``None`` if none qualify.

    ``embed_texts([query])[0]`` is searched against ``store``; only a top hit
    whose cosine similarity is ``>= threshold`` is accepted. Its id is then
    resolved via ``session`` into the ``KnowledgeChunk`` row and the ``answer``
    returned. An empty collection yields no hits and therefore ``None``.
    """
    vector = embedder.embed_texts([query])[0]
    hits = store.search(vector, top_k)
    if not hits:
        return None

    chunk_id, score = hits[0]
    if score < threshold:
        return None

    chunk = session.get(KnowledgeChunk, chunk_id)
    return chunk.answer if chunk else None


def default_retrieve(query: str) -> str | None:
    """Integration entry point: build real store/embedder/session from settings.

    Used by the chat tool path (``query_faq``) and Task 8 wiring. The collection
    is ensured before searching so a fresh data dir still degrades gracefully to
    ``None`` instead of raising.
    """
    settings = get_settings()
    store = MilvusStore(
        settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    store.ensure_collection()
    embedder = Embedder()

    with SessionLocal() as session:
        return retrieve(
            session,
            store,
            embedder,
            query,
            top_k=settings.retrieval_top_k,
            threshold=settings.retrieval_threshold,
        )
