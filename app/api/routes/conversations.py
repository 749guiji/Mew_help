"""会话侧栏接口:列用户会话 + 回载历史原文 + 新建空会话。"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.db.models import Conversation, Message

router = APIRouter()


class ConversationCreate(BaseModel):
    user_id: str

    @field_validator("user_id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/api/conversations")
def create_conversation(payload: ConversationCreate) -> dict:
    """新建一个空会话(供前端「新对话」立即落一条,而非发首条消息才建)。"""
    with SessionLocal() as s:
        conv = Conversation(user_id=payload.user_id, status="进行中")
        s.add(conv)
        s.commit()
        s.refresh(conv)
        return {"id": conv.id}


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
