"""会话侧栏只读接口:列用户会话 + 回载历史原文。"""

from __future__ import annotations

from fastapi import APIRouter

from app.db import SessionLocal
from app.db.models import Conversation, Message

router = APIRouter()


@router.get("/api/conversations")
def list_conversations(user_id: str) -> dict:
    """列该用户全部会话(新在前、首问预览、已摘要标记)。"""
    with SessionLocal() as s:
        convs = (
            s.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.id.desc())
            .all()
        )
        out = []
        for c in convs:
            first = (
                s.query(Message)
                .filter(Message.conversation_id == c.id, Message.role == "user")
                .order_by(Message.id)
                .first()
            )
            preview = (first.content[:50] if first and first.content else "")
            out.append(
                {"id": c.id, "preview": preview, "summarized": bool(c.summary)}
            )
        return {"conversations": out}


@router.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int) -> dict:
    """回载该会话的历史原文(user/assistant)。"""
    with SessionLocal() as s:
        msgs = (
            s.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id)
            .all()
        )
        return {
            "messages": [
                {"id": m.id, "role": m.role, "content": m.content} for m in msgs
            ]
        }
