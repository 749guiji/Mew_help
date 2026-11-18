"""Database-backed tools (FAQ lookup and ticket creation)."""

import json
import logging
import random
from datetime import datetime

from langchain_core.tools import tool

from app.core.config import get_settings
from app.db import SessionLocal
from app.db.models import Ticket
from app.kb.retrieval import default_retrieve
from app.services.generation_quality import build_citations, format_evidence
from app.tools.context import current_conversation_id

logger = logging.getLogger(__name__)


@tool
def query_faq(keyword: str, category: str | None = None) -> str:
    """查询售后政策、运费/邮费标准、退换货规则等常见问题(FAQ)的答案。当用户询问政策、规定、费用标准、规则等知识类问题时使用。

    返回 JSON 字符串：{"refused": bool, "evidence": str, "citations": list}。
    """
    try:
        chunks = default_retrieve(keyword)
    except Exception as e:
        logger.warning("query_faq failed: %s", e)
        chunks = []

    if not chunks or max(c.score for c in chunks) < get_settings().retrieval_threshold:
        return json.dumps(
            {"refused": True, "evidence": "", "citations": []}, ensure_ascii=False
        )

    return json.dumps(
        {
            "refused": False,
            "evidence": format_evidence(chunks),
            "citations": build_citations(chunks),
        },
        ensure_ascii=False,
    )


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
