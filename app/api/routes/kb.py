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
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from app.core.config import get_settings
from app.db import SessionLocal
from app.kb.chunker import Chunk, chunk_markdown
from app.kb.embedding import Embedder
from app.kb.ingest import insert_chunks, vectorize_pending
from app.kb.retrieval import _get_store, default_retrieve

router = APIRouter()

# kb.py is at app/api/routes/kb.py -> parents[3] is the repo root.
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"
KNOWLEDGE_DIR = Path("knowledge")


def _content_type_from_path(path: Path) -> str:
    """Derive content_type from a file name (same rule as ingest_docs)."""
    name = path.name.lower()
    if "faq" in name:
        return "faq"
    if "policy" in name or "退货" in name:
        return "policy"
    return "manual"


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

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.get("/kb")
def kb_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "kb.html")


@router.get("/api/kb/materials")
def list_materials() -> list[dict]:
    out = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        out.append(
            {
                "name": path.name,
                "content_type": _content_type_from_path(path),
                "chars": len(path.read_text(encoding="utf-8")),
            }
        )
    return out


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
        _content_type_from_path(target),
        max_chars=settings.chunk_max_chars,
        overlap=settings.chunk_overlap_chars,
    )
    return [asdict(c) for c in chunks]


@router.post("/api/kb/chunks")
def add_chunk(payload: KBChunkCreate) -> dict:
    with SessionLocal() as session:
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


@router.post("/api/kb/search")
def search(payload: KBSearchRequest) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "score": c.score,
            "section_path": c.section_path,
            "category": c.category,
            "answer": c.answer,
            "questions": c.questions,
        }
        for c in default_retrieve(payload.query)
    ]
