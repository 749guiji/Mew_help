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
