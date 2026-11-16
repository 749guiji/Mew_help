"""Database-backed tools (FAQ lookup and ticket creation)."""

import random
from datetime import datetime

from langchain_core.tools import tool

from app.db import SessionLocal
from app.db.models import Faq, Ticket
from app.tools.context import current_conversation_id


@tool
def query_faq(keyword: str) -> str:
    """查询售后政策、运费/邮费标准、退换货规则等常见问题(FAQ)的答案。当用户询问政策、规定、费用标准、规则等知识类问题时使用。"""
    with SessionLocal() as s:
        faq = s.query(Faq).filter(Faq.question.like(f"%{keyword}%")).first()
        return faq.answer if faq else "未查到"


@tool
def create_ticket(description: str, ticket_type: str) -> str:
    """当用户明确要求人工服务、投诉、或需要记录问题转人工处理时,创建人工工单。ticket_type 取:售后、投诉、咨询。"""
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
