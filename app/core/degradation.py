"""ch07 降级运行时接线:层1超预算上移锚点、层2超预算后台起摘要。"""

from __future__ import annotations

import logging

from app.core.budget import compute_budget
from app.core.config import get_settings
from app.core.summarizer import trigger_summary_if_needed
from app.db import SessionLocal
from app.db.models import Conversation, Message

logger = logging.getLogger("degradation")


def apply_degradation(
    conversation_id: int | None, layer1_chars: int, layer2_chars: int
) -> None:
    """拼装时调用:层1超预算上移 ``layer1_from`` 锚点;层2超预算后台起摘要。"""
    if not conversation_id:
        return
    budget = compute_budget(get_settings())
    if layer1_chars > budget.layer1 and budget.layer1 > 0:
        _move_layer1_anchor(conversation_id, layer1_chars, budget.layer1)
    trigger_summary_if_needed(conversation_id, layer2_chars, budget.layer2)


def _move_layer1_anchor(conversation_id: int, layer1_chars: int, layer1_budget: int) -> None:
    """层1超预算:把最老的层1消息降级进层2,上移 ``layer1_from_msg_id`` 直到层1装得下。"""
    with SessionLocal() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            return
        start_id = conv.layer1_from_msg_id or 0
        msgs = (
            s.query(Message)
            .filter(Message.conversation_id == conversation_id, Message.id > start_id)
            .order_by(Message.id)
            .all()
        )
        remaining = layer1_chars
        new_from = start_id
        for m in msgs:
            if remaining <= layer1_budget:
                break
            remaining -= len(m.content or "")
            new_from = m.id
        if new_from != start_id:
            logger.info(
                "degrade layer1 conv=%s %s→%s", conversation_id, start_id or "NULL", new_from
            )
            conv.layer1_from_msg_id = new_from
            s.commit()
