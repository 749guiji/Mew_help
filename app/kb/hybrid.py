"""Hybrid retrieval orchestration: four strategies + the understanding pipeline.

The four strategy functions are the evaluation surface (dense / sparse /
hybrid-RRF / hybrid+rerank). Each one searches Milvus for candidate hits and
then backfills ``answer`` / ``questions`` / ``section_path`` / ``category`` /
``content_type`` from the authoritative ``knowledge_chunks`` row in MySQL, so
callers always receive the source-of-truth text rather than a copy cached in
the vector store.

``retrieve_with_understanding`` is the production path: it rewrites a
colloquial question, expands synonyms (for the BM25 side only), and then runs
the hybrid+rerank strategy with an optional category filter inferred by the
rewrite step.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.db.models import KnowledgeChunk
from app.kb.milvus_store import Hit
from app.kb.query_understanding import expand_synonyms, rewrite_query


def _escape_filter_string(value: str) -> str:
    """Escape ``value`` for a Milvus double-quoted string literal."""
    return value.replace('"', '\\"')


@dataclass
class ScoredChunk:
    """One retrieved chunk with its rank score and MySQL backfilled fields."""

    chunk_id: int
    score: float
    section_path: str | None
    category: str
    content_type: str
    answer: str
    questions: str


def _backfill(hits: list[Hit], session) -> list[ScoredChunk]:
    """Resolve each hit's id into a ``KnowledgeChunk`` row (MySQL backfill)."""
    chunks: list[ScoredChunk] = []
    for hit in hits:
        row = session.get(KnowledgeChunk, hit.id)
        if row is None:
            continue
        chunks.append(
            ScoredChunk(
                chunk_id=row.id,
                score=hit.score,
                section_path=row.section_path,
                category=row.category,
                content_type=row.content_type or "",
                answer=row.answer,
                questions=row.questions,
            )
        )
    return chunks


def dense_retrieve(query, *, session, store, embedder, top_k=10, filter=None):
    """Cosine (dense) retrieval, ranked by similarity score."""
    vector = embedder.embed_texts([query])[0]
    hits = store.search_dense(vector, top_k=top_k, filter=filter)
    return _backfill(hits, session)


def sparse_retrieve(query, *, session, store, top_k=10, filter=None):
    """BM25 (sparse) full-text retrieval, ranked by keyword score."""
    hits = store.search_sparse(query, top_k=top_k, filter=filter)
    return _backfill(hits, session)


def hybrid_retrieve(query, *, session, store, embedder, top_k=10, filter=None):
    """Dense + sparse Reciprocal Rank Fusion, without a rerank pass."""
    vector = embedder.embed_texts([query])[0]
    hits = store.hybrid_search(vector, query, top_k=top_k, filter=filter)
    return _backfill(hits, session)


def hybrid_rerank_retrieve(
    query,
    *,
    session,
    store,
    embedder,
    reranker,
    top_k=10,
    dense_top_k=50,
    sparse_top_k=50,
    filter=None,
    sparse_text: str | None = None,
):
    """RRF over top-dense/sparse candidates, then rerank to ``top_k``.

    The dense embedding and the reranker both use ``query`` (the standard
    question); ``sparse_text`` overrides the BM25 side so callers can feed an
    expanded form without changing the dense/rerank semantics.
    """
    vector = embedder.embed_texts([query])[0]
    sparse_q = query if sparse_text is None else sparse_text
    # RRF fuses the two ranked lists inside Milvus; fetching ``max(dense_top_k,
    # sparse_top_k)`` from each side widens the candidate pool before reranking.
    candidate_k = max(dense_top_k, sparse_top_k)
    hits = store.hybrid_search(vector, sparse_q, top_k=candidate_k, filter=filter)
    chunks = _backfill(hits, session)
    if not chunks:
        return []

    scores = reranker.rerank(query, [c.answer for c in chunks])
    scored = [
        ScoredChunk(
            chunk_id=c.chunk_id,
            score=float(score),
            section_path=c.section_path,
            category=c.category,
            content_type=c.content_type,
            answer=c.answer,
            questions=c.questions,
        )
        for c, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def retrieve_with_understanding(
    query,
    *,
    session,
    store,
    embedder,
    reranker,
    rewrite_llm,
    top_k=10,
    filter=None,
) -> list[ScoredChunk]:
    """Rewrite -> synonym expansion (BM25 only) -> hybrid+rerank -> filter."""
    u = rewrite_query(rewrite_llm, query)
    sparse_q = expand_synonyms(u.standard_question)
    cat_filter = filter or (
        f'category == "{_escape_filter_string(u.category)}"' if u.category else None
    )
    return hybrid_rerank_retrieve(
        u.standard_question,
        session=session,
        store=store,
        embedder=embedder,
        reranker=reranker,
        top_k=top_k,
        filter=cat_filter,
        sparse_text=sparse_q,
    )
