"""动作端点:订单选择器续跑(resume)+ 退款单落库(复用 tickets 表)。"""

from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.db.models import Conversation, Ticket

router = APIRouter()


class ResumeRequest(BaseModel):
    conversation_id: int
    order_id: str | None = None
    confirm: bool | None = None

    @field_validator("order_id")
    @classmethod
    def _not_blank(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class CreateRefundRequest(BaseModel):
    session_id: str
    order_id: str
    reason: Literal["七天无理由", "质量问题", "发错货", "不想要了", "其他"]

    @field_validator("session_id", "order_id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/api/actions/resume")
async def resume(payload: ResumeRequest, request: Request) -> StreamingResponse:
    """续跑暂停流(SSE,事件与 /chat 同构):订单选择(order_id)或建工单确认(confirm)。"""
    svc = request.app.state.chat_service
    resume_value = (
        {"confirm": payload.confirm} if payload.confirm is not None else payload.order_id
    )

    async def gen():
        async for event in svc.stream_resume(payload.conversation_id, resume_value):
            yield f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/api/actions/create-refund")
def create_refund(payload: CreateRefundRequest) -> dict:
    """退款表单提交:写 tickets(ticket_type='退款'),描述带订单号 + 固定类目原因。"""
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

        ticket_no = f"T{datetime.now():%Y%m%d%H%M%S}{random.randint(0, 999):03d}"
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conv.id,
                description=f"退款申请 订单号={payload.order_id} 原因={payload.reason}",
                ticket_type="退款",
            )
        )
        s.commit()
        return {"ticket_no": ticket_no, "status": "退款申请已提交"}
