"""Admin aggregate homepage: a shell of module cards + a shared nav.

Each card is built by an independent function so a missing dependency only
degrades that card (``error`` is set on the card, others keep working).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.db import SessionLocal
from app.db.models import Conversation, KnowledgeChunk, ReviewQueue, Ticket
from app.kb.retrieval import _get_store

router = APIRouter()

# admin.py is at app/api/routes/admin.py -> parents[3] is the repo root.
STATIC_DIR = Path(__file__).resolve().parents[3] / "static"


def _kb_card() -> dict:
    try:
        with SessionLocal() as session:
            chunks = session.query(KnowledgeChunk).count()
            pending = (
                session.query(KnowledgeChunk)
                .filter(KnowledgeChunk.vectorize_status == "pending")
                .count()
            )
        milvus = _get_store().count()
        return {
            "key": "kb",
            "title": "知识库",
            "metrics": {"块数": chunks, "待向量化": pending, "Milvus": milvus},
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - a missing dependency only affects this card
        return {"key": "kb", "title": "知识库", "metrics": {}, "error": str(exc)}


def _mcp_card() -> dict:
    try:
        import asyncio

        from app.mcp.client import mcp_server_status

        status = asyncio.run(mcp_server_status())
        labels = {"logistics": "物流", "aftersale": "售后"}
        return {
            "key": "mcp",
            "title": "MCP 服务",
            "metrics": {
                labels.get(name, name): "在线" if ok else "离线"
                for name, ok in status.items()
            },
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"key": "mcp", "title": "MCP 服务", "metrics": {}, "error": str(exc)}


def _stats_card() -> dict:
    try:
        with SessionLocal() as s:
            convs = s.query(Conversation).count()
            pending = (
                s.query(ReviewQueue)
                .filter(ReviewQueue.review_status == "待审")
                .count()
            )
            tickets = s.query(Ticket).count()
        return {
            "key": "stats",
            "title": "运营概览",
            "metrics": {"会话数": convs, "待审队列": pending, "工单数": tickets},
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"key": "stats", "title": "运营概览", "metrics": {}, "error": str(exc)}


CARD_BUILDERS = [_stats_card, _kb_card, _mcp_card]


@router.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@router.get("/api/admin/overview")
def overview() -> dict:
    return {"cards": [build() for build in CARD_BUILDERS]}
