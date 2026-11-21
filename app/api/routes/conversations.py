"""会话侧栏接口:列/新建/重命名/置顶/删除会话 + 回载历史原文。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.db.models import (
    Conversation,
    ConversationSummary,
    LowConfidenceQuestion,
    Message,
    Ticket,
)

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


class ConversationUpdate(BaseModel):
    title: str | None = None
    pinned: bool | None = None


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
    """列该用户全部会话(置顶在前、其余新在前;首问预览 + 标题 + 置顶标记)。"""
    with SessionLocal() as s:
        convs = (
            s.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.pinned.desc(), Conversation.id.desc())
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
                {
                    "id": c.id,
                    "title": c.title,
                    "pinned": bool(c.pinned),
                    "preview": preview,
                    "summarized": bool(c.summary),
                }
            )
        return {"conversations": out}


@router.patch("/api/conversations/{conversation_id}")
def update_conversation(conversation_id: int, payload: ConversationUpdate) -> dict:
    """重命名(title)或置顶/取消置顶(pinned)。"""
    with SessionLocal() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if payload.title is not None:
            conv.title = payload.title.strip() or None
        if payload.pinned is not None:
            conv.pinned = payload.pinned
        s.commit()
        return {"id": conv.id, "title": conv.title, "pinned": bool(conv.pinned)}


@router.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int) -> dict:
    """删除会话,级联清掉其消息/摘要/工单/低置信问题(审计留痕无外键、保留)。"""
    with SessionLocal() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        s.query(Message).filter(Message.conversation_id == conversation_id).delete()
        s.query(ConversationSummary).filter(
            ConversationSummary.conversation_id == conversation_id
        ).delete()
        s.query(Ticket).filter(Ticket.conversation_id == conversation_id).delete()
        s.query(LowConfidenceQuestion).filter(
            LowConfidenceQuestion.conversation_id == conversation_id
        ).delete()
        s.delete(conv)
        s.commit()
        return {"ok": True}


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
