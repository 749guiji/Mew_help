"""LangGraph graph state (ch05)."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    """State threaded through the orchestration graph.

    ``messages`` carries the multi-turn conversation (``add_messages`` reducer);
    the rest are per-turn fields set by the deterministic nodes.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    raw_query: str
    resolved_query: str
    user_id: str
    conversation_id: int | None
    intent: str
    route: str
    retrieval_chunks: list
    retrieval_score: float | None
    confidence_pass: bool | None
    handoff_suggestions: list[str]
    suggested_actions: list
    visited: Annotated[list[str], operator.add]
    # ch06:退款/售后子流程(interrupt 承载槽位,无需 pending_slot)
    order_id: str | None
    order_data: dict | None
    intent_confidence: float | None
    # ch08:待前端确认的建工单预览(agent 记下 → confirm_ticket 节点 interrupt)
    pending_ticket: dict | None
    # ch08:本轮工具执行结果 tool_call_id → done/error(前端徽章区分成败)
    tool_status: dict[str, str]
    # ch09:知识类闸算出的 evidence_confidence(confidence + signals)
    evidence_confidence: dict | None
