"""ch07 后台摘要任务:层2 攒到超预算就起一次,压成新段追加进 conversation_summaries。

只追加、不回头重写;旧梗概只作背景给模型看、不参与合并(一个事实只经历一次有损压缩)。
触发看用量不数条数(工具结果不落消息表,数条数看不见它涨)。
"""

from __future__ import annotations

import logging
import threading
import time

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import func

from app.db import SessionLocal
from app.db.models import Conversation, ConversationSummary, Message
from app.prompts.system import SUMMARY_SYSTEM_PROMPT

logger = logging.getLogger("summarizer")

# 每会话一把去重锁(键 = conversation_id)
_LOCKS: dict[int, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(conversation_id: int) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(conversation_id, threading.Lock())


def _summarize(llm, turns: list[str]) -> str:
    """调模型把一段对话压成摘要;异常/空 → 返回空串(调用方跳过)。"""
    chain = ChatPromptTemplate.from_messages(
        [("system", SUMMARY_SYSTEM_PROMPT), ("human", "{conversation}")]
    ) | llm
    result = chain.invoke({"conversation": "\n".join(turns)})
    text = getattr(result, "content", str(result)).strip()
    return text


def summarize_layer2(conversation_id: int, *, llm=None) -> None:
    """读层2消息 → 摘要 → 追加 conversation_summaries → 更新 summary + summary_upto=layer1_from。"""
    from app.core.config import get_settings
    from app.core.llm import get_llm

    t0 = time.time()
    logger.info("summary start conv=%s", conversation_id)
    with SessionLocal() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            logger.info("summary fail conv=%s(会话不存在)", conversation_id)
            return
        if not conv.layer1_from_msg_id:
            logger.info("summary skip conv=%s(无 layer1_from 锚点)", conversation_id)
            return
        upto = conv.summary_upto_msg_id or 0
        msgs = (
            s.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.id > upto,
                Message.id <= conv.layer1_from_msg_id,
            )
            .order_by(Message.id)
            .all()
        )
        if not msgs:
            logger.info("summary skip conv=%s(层2空)", conversation_id)
            return

        turns = [f"{'用户' if m.role == 'user' else '客服'}:{m.content}" for m in msgs]
        text = _summarize(llm or get_llm(get_settings()), turns)
        if not text:
            logger.info("summary fail conv=%s(摘要为空)", conversation_id)
            return

        max_seq = (
            s.query(func.max(ConversationSummary.seq))
            .filter(ConversationSummary.conversation_id == conversation_id)
            .scalar()
            or 0
        )
        s.add(
            ConversationSummary(
                conversation_id=conversation_id,
                seq=max_seq + 1,
                from_msg_id=msgs[0].id,
                upto_msg_id=msgs[-1].id,
                content=text,
            )
        )
        conv.summary = (conv.summary + "\n" + text) if conv.summary else text
        conv.summary_upto_msg_id = conv.layer1_from_msg_id
        s.commit()
        logger.info(
            "summary done conv=%s seg覆盖[%d,%d] 耗时%.2fs",
            conversation_id, msgs[0].id, msgs[-1].id, time.time() - t0,
        )


def trigger_summary_if_needed(conversation_id: int, layer2_usage: int, layer2_budget: int) -> None:
    """层2用量超预算就后台起摘要(非阻塞;已在跑则跳过)。"""
    if layer2_usage <= layer2_budget:
        return
    lock = _lock_for(conversation_id)
    if not lock.acquire(blocking=False):
        logger.info("summary skip conv=%s(已在跑)", conversation_id)
        return
    logger.info("summary trigger conv=%s layer2=%d>%d", conversation_id, layer2_usage, layer2_budget)

    def _run() -> None:
        try:
            summarize_layer2(conversation_id)
        except Exception as exc:  # noqa: BLE001 - 留痕,不阻塞
            logger.info("summary fail conv=%s(%s)", conversation_id, exc)
        finally:
            lock.release()

    threading.Thread(target=_run, daemon=True).start()
