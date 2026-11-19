"""Graph nodes (ch05).

Deterministic + LLM nodes: coreference passthrough, intent recognition,
knowledge retrieval, confidence gate, fixed replies, and logging. The main
Agent node lives in :mod:`app.graph.agent`.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.prompts.system import INTENT_SYSTEM_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


def resolve_coref(query: str) -> str:
    """指代消解:本章原样透传(正式版后续再做)。"""
    return query


def classify_intent(llm, query: str) -> str:
    """Classify ``query`` into one of seven intents via plain text + match.

    qwen3:8b 的 function-calling 结构化输出对这类单字段 schema 偶发返回 None
    (known issue),故改走纯文本:让模型只输出类别名,再用子串匹配兜底「闲聊」。
    """
    chain = ChatPromptTemplate.from_messages(
        [("system", INTENT_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm
    result = chain.invoke({"input": query})
    text = getattr(result, "content", str(result)).strip()
    for intent in INTENTS:
        if intent in text:
            return intent
    return "闲聊"


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
