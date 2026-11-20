"""Graph nodes (ch05).

Deterministic + LLM nodes: coreference passthrough, intent recognition,
knowledge retrieval, confidence gate, fixed replies, and logging. The main
Agent node lives in :mod:`app.graph.agent`.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from app.graph.json_utils import parse_json_object
from app.prompts.system import (
    COREF_REWRITE_SYSTEM_PROMPT,
    EXPAND_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")

# 合法意图 = 七类 + 「其他」兜底;越界的类别名一律归「其他」,防止 route 崩溃。
VALID_INTENTS = INTENTS + ("其他",)


@dataclass
class IntentResult:
    """意图识别结果:intent 为七类之一或「其他」,confidence ∈ [0, 1]."""

    intent: str
    confidence: float

_COREF_HISTORY_MAX_CHARS = 2000


def _build_coref_context(query: str, history: list[str]) -> str:
    """把历史 + 当前问题拼成给 LLM 的上下文,超出字符预算时丢最旧消息。"""
    lines: list[str] = []
    total = 0
    for msg in reversed(history):
        total += len(msg)
        if total > _COREF_HISTORY_MAX_CHARS:
            break
        lines.append(msg)
    lines.reverse()
    history_text = "\n".join(f"- {m}" for m in lines) if lines else "(无)"
    return f"对话历史：\n{history_text}\n\n当前问题：{query}"


def _clean_single_line(text: str) -> str:
    """去掉代码围栏、配对引号与首尾空白,只留单句。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


def resolve_coref(llm, query: str, history: list[str]) -> str:
    """指代消解 + 改写:结合历史把 query 补全成独立完整问句(单句,非 JSON)。

    ``llm`` 是 Runnable;``history`` 为已提取的最近若干轮用户/助手消息文本
    (旧→新)。空输出/异常一律回退 ``query``。
    """
    chain = ChatPromptTemplate.from_messages(
        [("system", COREF_REWRITE_SYSTEM_PROMPT), ("human", "{context}")]
    ) | llm
    result = chain.invoke({"context": _build_coref_context(query, history)})
    text = getattr(result, "content", str(result)).strip()
    return _clean_single_line(text) or query


def classify_intent(llm, query: str, history: str = "") -> IntentResult:
    """Classify ``query`` into one of seven intents (+「其他」) via JSON prompt.

    结合最近几轮历史判当前意图(应对物流→退款→物流的漂移)。强制 JSON + 稳健解析:
    解析失败或字段缺失 → ``IntentResult("其他", 0.0)``。
    """
    context = f"最近对话(可空):\n{history or '(无)'}\n\n当前问题:{query}"
    chain = ChatPromptTemplate.from_messages(
        [("system", INTENT_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm
    result = chain.invoke({"input": context})
    text = getattr(result, "content", str(result)).strip()
    data = parse_json_object(text)
    if not data:
        return IntentResult("其他", 0.0)
    intent = data.get("intent")
    confidence = data.get("confidence")
    if not isinstance(intent, str) or not isinstance(confidence, (int, float)):
        return IntentResult("其他", 0.0)
    if intent not in VALID_INTENTS:
        return IntentResult("其他", 0.0)
    return IntentResult(intent, float(confidence))


def expand_queries(llm, query: str) -> list[str]:
    """把一句完整问法扩成多条侧重点不同的检索查询(强制 JSON ``queries[]``)。

    解析失败/空/queries 非字符串列表 → 回退 ``[query]``。
    """
    chain = ChatPromptTemplate.from_messages(
        [("system", EXPAND_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm
    result = chain.invoke({"input": query})
    text = getattr(result, "content", str(result)).strip()
    data = parse_json_object(text)
    queries = data.get("queries") if data else None
    if not isinstance(queries, list) or not queries:
        return [query]
    cleaned = [q for q in queries if isinstance(q, str) and q.strip()]
    return cleaned or [query]


def gate_pass(score: float | None, threshold: float) -> bool:
    """Confidence gate: pass only when a retrieval score meets the threshold."""
    return score is not None and score >= threshold


def retrieve_node(state) -> dict:
    """知识检索:调 ch04 的 default_retrieve 取 Top-K,填 retrieval_chunks/score."""
    from app.kb.retrieval import default_retrieve

    chunks = default_retrieve(state["resolved_query"])
    score = max((c.score for c in chunks), default=None) if chunks else None
    return {
        "retrieval_chunks": chunks,
        "retrieval_score": score,
        "visited": ["retrieve"],
    }


def gate_node(state) -> dict:
    """置信度闸:按检索得分阈值判定是否进 Agent。"""
    from app.core.config import get_settings

    passed = gate_pass(state.get("retrieval_score"), get_settings().retrieval_threshold)
    return {"confidence_pass": passed, "visited": ["gate"]}


def fallback_node(state) -> dict:
    """知识类证据不足:回兜底话术 + 记 low_confidence_questions(数据飞轮预留)。"""
    from langchain_core.messages import AIMessage

    from app.core.config import get_settings
    from app.db import SessionLocal
    from app.services.generation_quality import log_low_confidence

    try:
        with SessionLocal() as s:
            log_low_confidence(
                s,
                conversation_id=None,
                raw_question=state.get("resolved_query", ""),
                source="retrieval_low_conf",
                reason=f"检索得分低于阈值 {get_settings().retrieval_threshold}",
            )
    except Exception:  # noqa: BLE001 - 记录失败不阻塞主流程
        pass
    return {
        "messages": [AIMessage(content="抱歉，我目前的知识库中没有足够信息回答您的问题，已为您记录并转人工处理。")],
        "visited": ["fallback"],
    }


def complaint_node(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个建议(不自动执行)。"""
    from langchain_core.messages import AIMessage

    return {
        "messages": [AIMessage(content="非常抱歉给您带来了不好的体验。您可以点击下方按钮转接人工客服，或创建工单由我们跟进处理。")],
        "handoff_suggestions": ["转人工", "建工单"],
        "visited": ["complaint"],
    }


def chitchat_node(state) -> dict:
    """闲聊:固定话术,不花模型调用。"""
    from langchain_core.messages import AIMessage

    return {
        "messages": [AIMessage(content="您好呀，我是客服助手，请问有什么可以帮您的？")],
        "visited": ["chitchat"],
    }


def log_node(state) -> dict:
    """日志记录:把本次路径 + 意图 + 分流写进日志(供验收#1)。"""
    import logging

    logging.getLogger("graph").info(
        "visited=%s intent=%s route=%s",
        state.get("visited"),
        state.get("intent"),
        state.get("route"),
    )
    return {"visited": ["log"]}
