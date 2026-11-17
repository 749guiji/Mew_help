"""Dual-write ingest: MySQL ``knowledge_chunks`` as source of truth + Milvus.

This module wires the offline build path together: :class:`Chunk` objects from
:mod:`app.kb.chunker` are written into ``knowledge_chunks`` in a ``pending``
state, then embedded and upserted into Milvus, after which the row is backfilled
to ``done`` with its ``vector_id``. Because the row only flips to ``done`` after
a successful Milvus write, re-running ``vectorize_pending`` is idempotent and
picks up exactly the rows that were left behind by an interrupted build.
"""

from __future__ import annotations

from app.db.models import Faq, KnowledgeChunk
from app.kb.chunker import Chunk, chunk_markdown


def _to_row(chunk: Chunk) -> KnowledgeChunk:
    return KnowledgeChunk(
        category=chunk.category,
        questions=chunk.questions,
        answer=chunk.answer,
        section_path=chunk.section_path,
        content_type=chunk.content_type,
        is_key_clause=int(chunk.is_key_clause),
        vectorize_status="pending",
    )


def insert_chunks(session, chunks: list[Chunk]) -> list[int]:
    """Persist chunks as pending ``KnowledgeChunk`` rows, in input order.

    ``session.add_all`` + ``flush()`` assigns PKs so the ordered
    ``prev_chunk_id``/``next_chunk_id`` pointers can be backfilled before the
    single ``commit``. Returns the row ids in the same order as ``chunks``.
    """
    rows = [_to_row(c) for c in chunks]
    session.add_all(rows)
    session.flush()

    ids = [row.id for row in rows]
    for i, row in enumerate(rows):
        if i > 0:
            row.prev_chunk_id = rows[i - 1].id
        if i < len(rows) - 1:
            row.next_chunk_id = rows[i + 1].id

    session.commit()
    return ids


def _vectorize_text(row: KnowledgeChunk) -> str:
    return f"{row.category}\n{row.questions}\n{row.answer}"


def vectorize_pending(session, store, embedder) -> int:
    """Embed + upsert every ``pending`` row, then backfill ``done``.

    Returns the number of rows vectorized. Rows already ``done`` are skipped, so
    calling this again is a no-op (idempotent reconcile).
    """
    pending = (
        session.query(KnowledgeChunk)
        .filter(KnowledgeChunk.vectorize_status == "pending")
        .order_by(KnowledgeChunk.id)
        .all()
    )
    if not pending:
        return 0

    texts = [_vectorize_text(row) for row in pending]
    vectors = embedder.embed_texts(texts)
    for vector in vectors:
        assert len(vector) == embedder.dim, (
            f"embedding dimension mismatch: expected {embedder.dim}, got {len(vector)}"
        )

    store.upsert([(row.id, vector) for row, vector in zip(pending, vectors)])

    for row in pending:
        row.vectorize_status = "done"
        row.vector_id = str(row.id)
    session.commit()
    return len(pending)


def ingest_document(
    session,
    store,
    embedder,
    text,
    content_type,
    *,
    max_chars,
    overlap,
    source: str | None = None,
) -> list[int]:
    """Chunk ``text``, insert it, then vectorize all pending rows."""
    chunks = chunk_markdown(text, content_type, max_chars=max_chars, overlap=overlap)
    ids = insert_chunks(session, chunks)
    vectorize_pending(session, store, embedder)
    return ids


def migrate_faq(session, store, embedder) -> int:
    """Convert every ``faq`` table row into a ``content_type='faq'`` chunk."""
    faqs = session.query(Faq).all()
    if not faqs:
        return 0

    chunks = [
        Chunk(
            category=faq.category,
            questions=faq.question,
            answer=faq.answer,
            section_path=None,
            content_type="faq",
            is_key_clause=False,
        )
        for faq in faqs
    ]
    insert_chunks(session, chunks)
    vectorize_pending(session, store, embedder)
    return len(chunks)
