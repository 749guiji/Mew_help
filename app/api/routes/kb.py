"""KB admin page backend: material listing, chunk preview, manual entry,
pending vectorization, and retrieval self-test.

These endpoints reuse the process-cached Milvus store / embedder / reranker
(from :mod:`app.kb.retrieval`), so they must run inside the uvicorn process —
not alongside a separate ``ingest_docs`` CLI (Milvus Lite is single-process
per data dir).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.db import SessionLocal
from app.db.models import KnowledgeChunk, QaExtractionStaging
from app.kb.chunker import Chunk, chunk_markdown
from app.kb.embedding import Embedder
from app.kb.hybrid import (
    dense_retrieve,
    hybrid_rerank_retrieve,
    hybrid_retrieve,
    sparse_retrieve,
)
from app.kb.ingest import (
    chunk_fingerprint,
    find_duplicate_chunk,
    insert_chunks,
    vectorize_pending,
)
from app.kb.reranker import Reranker
from app.kb.retrieval import _get_store
from app.kb.sources import content_type_from_path, list_kb_materials

router = APIRouter()

# kb.py is at app/api/routes/kb.py -> parents[3] is the repo root.
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"
KNOWLEDGE_DIR = Path("knowledge")


class KBChunkCreate(BaseModel):
    content_type: str
    category: str
    questions: str
    answer: str

    @field_validator("questions", "answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class KBSearchRequest(BaseModel):
    query: str
    strategy: str = "hybrid_rerank"

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


STRATEGIES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@router.get("/kb")
def kb_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "kb.html")


@router.get("/api/kb/materials")
def list_materials() -> list[dict]:
    return [
        {"name": m.name, "content_type": m.content_type, "chars": m.chars}
        for m in list_kb_materials()
    ]


@router.get("/api/kb/preview")
def preview(path: str) -> list[dict]:
    # basename() pins the file inside knowledge/, so ../ traversal can't escape.
    target = KNOWLEDGE_DIR / Path(path).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="material not found")

    settings = get_settings()
    text = target.read_text(encoding="utf-8")
    chunks = chunk_markdown(
        text,
        content_type_from_path(target),
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap_chars,
    )
    return [asdict(c) for c in chunks]


@router.post("/api/kb/chunks")
def add_chunk(payload: KBChunkCreate) -> dict:
    fingerprint = chunk_fingerprint(payload.questions, payload.answer)
    with SessionLocal() as session:
        existing = find_duplicate_chunk(session, fingerprint)
        if existing is not None:
            return JSONResponse(
                status_code=409, content={"duplicate": True, "id": existing}
            )
        ids = insert_chunks(
            session,
            [
                Chunk(
                    category=payload.category,
                    questions=payload.questions,
                    answer=payload.answer,
                    section_path=None,
                    content_type=payload.content_type,
                    is_key_clause=False,
                )
            ],
        )
    return {"id": ids[0]}


@router.post("/api/kb/vectorize")
def vectorize() -> dict:
    settings = get_settings()
    store = _get_store()
    embedder = Embedder(dim=settings.embedding_dim)
    with SessionLocal() as session:
        n = vectorize_pending(session, store, embedder)
    return {"vectorized": n}


def _chunk_dict(c) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "score": c.score,
        "section_path": c.section_path,
        "category": c.category,
        "answer": c.answer,
        "questions": c.questions,
    }


@router.post("/api/kb/search")
def search(payload: KBSearchRequest) -> list[dict]:
    if payload.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=422, detail=f"unknown strategy: {payload.strategy}"
        )
    settings = get_settings()
    store = _get_store()
    embedder = Embedder(dim=settings.embedding_dim)
    with SessionLocal() as session:
        if payload.strategy == "dense":
            chunks = dense_retrieve(
                payload.query, session=session, store=store, embedder=embedder, top_k=10
            )
        elif payload.strategy == "sparse":
            chunks = sparse_retrieve(payload.query, session=session, store=store, top_k=10)
        elif payload.strategy == "hybrid":
            chunks = hybrid_retrieve(
                payload.query, session=session, store=store, embedder=embedder, top_k=10
            )
        else:  # hybrid_rerank
            reranker = Reranker(
                model_name=settings.rerank_model, use_fp16=settings.rerank_use_fp16
            )
            chunks = hybrid_rerank_retrieve(
                payload.query,
                session=session,
                store=store,
                embedder=embedder,
                reranker=reranker,
                top_k=10,
            )
    return [_chunk_dict(c) for c in chunks]


@router.get("/api/kb/overview")
def overview() -> dict:
    with SessionLocal() as session:
        chunks = session.query(KnowledgeChunk).count()
        pending = (
            session.query(KnowledgeChunk)
            .filter(KnowledgeChunk.vectorize_status == "pending")
            .count()
        )
        key_clauses = (
            session.query(KnowledgeChunk)
            .filter(KnowledgeChunk.is_key_clause == 1)
            .count()
        )
    done = chunks - pending
    milvus = None
    try:
        milvus = _get_store().count()
    except Exception:  # noqa: BLE001 - Milvus unreadable is a state, not an error
        milvus = None
    consistent = (done == milvus) if milvus is not None else None
    return {
        "chunks": chunks,
        "pending": pending,
        "done": done,
        "milvus": milvus,
        "key_clauses": key_clauses,
        "consistent": consistent,
    }


@router.get("/api/kb/staging")
def staging() -> dict:
    with SessionLocal() as session:
        counts = {
            status: session.query(QaExtractionStaging)
            .filter(QaExtractionStaging.status == status)
            .count()
            for status in ("extracted", "kept", "discarded")
        }
        rows = (
            session.query(QaExtractionStaging)
            .order_by(QaExtractionStaging.id.desc())
            .limit(50)
            .all()
        )
    return {
        **counts,
        "rows": [
            {
                "id": r.id,
                "batch_no": r.batch_no,
                "source_ref": r.source_ref,
                "question": r.question,
                "answer": r.answer,
                "status": r.status,
            }
            for r in rows
        ],
    }


@router.get("/api/kb/recent")
def recent(limit: int = 20) -> list[dict]:
    with SessionLocal() as session:
        rows = (
            session.query(KnowledgeChunk)
            .order_by(KnowledgeChunk.id.desc())
            .limit(limit)
            .all()
        )
    return [
        {
            "id": r.id,
            "category": r.category,
            "questions": r.questions,
            "answer": r.answer,
            "content_type": r.content_type,
            "vectorize_status": r.vectorize_status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
