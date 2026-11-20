"""退款单 API:前端退款表单直调写 refund_requests 表(不经过 LLM/Agent)。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.db.models import Conversation, RefundRequest

router = APIRouter()

RefundReason = Literal["7天无理由", "质量问题", "发错货", "少件", "与描述不符", "其他"]


class RefundCreate(BaseModel):
    session_id: str
    order_id: str
    reason: RefundReason

    @field_validator("session_id", "order_id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/api/refunds")
def create_refund(payload: RefundCreate) -> dict:
    """按 session_id 解出 conversation_id,写一条 refund_requests 记录。"""
    with SessionLocal() as s:
        conv = (
            s.query(Conversation)
            .filter(Conversation.user_id == payload.session_id)
            .first()
        )
        if conv is None:
            conv = Conversation(user_id=payload.session_id, status="进行中")
            s.add(conv)
            s.commit()
            s.refresh(conv)

        refund = RefundRequest(
            conversation_id=conv.id,
            order_id=payload.order_id,
            reason=payload.reason,
        )
        s.add(refund)
        s.commit()
        s.refresh(refund)
        return {"refund_id": refund.id}
