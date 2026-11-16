"""Database-backed tools (FAQ lookup and ticket creation)."""

import random
from datetime import datetime

from langchain_core.tools import tool

from app.db import SessionLocal
from app.db.models import Faq, Ticket
from app.tools.context import current_conversation_id


@tool
def query_faq(keyword: str) -> str:
    """根据关键词查询常见问题解答。"""
    with SessionLocal() as s:
        faq = s.query(Faq).filter(Faq.question.like(f"%{keyword}%")).first()
        return faq.answer if faq else "未查到"


@tool
def create_ticket(description: str, ticket_type: str) -> str:
    """根据问题描述创建工单，返回工单号。"""
    ticket_no = f"T{datetime.now():%Y%m%d%H%M%S}{random.randint(0, 999):03d}"
    ticket = Ticket(
        ticket_no=ticket_no,
        conversation_id=current_conversation_id.get(),
        description=description,
        ticket_type=ticket_type,
        status="待处理",
    )
    with SessionLocal() as s:
        s.add(ticket)
        s.commit()
    return ticket_no
