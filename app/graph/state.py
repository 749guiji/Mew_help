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
    intent: str
    route: str
    retrieval_chunks: list
    retrieval_score: float | None
    confidence_pass: bool | None
    handoff_suggestions: list[str]
    visited: Annotated[list[str], operator.add]
