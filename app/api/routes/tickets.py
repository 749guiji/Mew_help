"""建工单 API:前端「建工单」按钮直调写 tickets 表(不经过 LLM/Agent)。"""

from __future__ import annotations

import random
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from app.db import SessionLocal
from app.db.models import Conversation, Ticket

router = APIRouter()


class TicketCreate(BaseModel):
    session_id: str
    description: str
    ticket_type: str

    @field_validator("description", "ticket_type")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


@router.post("/api/tickets")
def create_ticket(payload: TicketCreate) -> dict:
    """按 session_id 解出 conversation_id,写一条 tickets 记录。"""
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
                description=payload.description,
                ticket_type=payload.ticket_type,
                status="待处理",
            )
        )
        s.commit()
        return {"ticket_no": ticket_no}
