"""Assemble the ch06 LangGraph orchestration graph (5 outlets + interrupt)."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import agent, nodes
from app.graph.router import route_from_intent
from app.graph.state import GraphState


def _reset_node(state: GraphState) -> dict:
    """清空上一轮的轮内字段,避免 checkpointer 里的旧值漏到本轮。

    interrupt/resume 不经过 reset(resume 从被中断节点续跑),所以这里清掉
    order_id/order_data 只影响「新一轮」,不影响同轮内的 resume 续跑。
    """
    return {
        "retrieval_chunks": [],
        "retrieval_score": None,
        "confidence_pass": None,
        "handoff_suggestions": [],
        "suggested_actions": [],
        "order_data": None,
        "order_id": None,
        "intent_confidence": None,
    }


def _history_text(state: GraphState) -> str:
    """coref/意图共用的历史:摘要行 + 最近几轮滑窗(每轮必打 history_ctx)。"""
    from app.graph.context import assemble_history_context
    from app.graph.context_meta import load_context_meta

    summary_text, _, _ = load_context_meta(state.get("conversation_id"))
    return assemble_history_context(state.get("messages", []), summary_text=summary_text)


def _coref_node(state: GraphState) -> dict:
    from app.core.config import get_settings
    from app.core.llm import get_llm

    history = [line for line in _history_text(state).split("\n") if line]
    resolved = nodes.resolve_coref(get_llm(get_settings()), state["raw_query"], history)
    return {"resolved_query": resolved, "visited": ["coref"]}


def _intent_node(state: GraphState) -> dict:
    from langchain_openai import ChatOpenAI

    from app.core.config import get_settings

    settings = get_settings()
    query = state["resolved_query"]

    # 意图识别统一走够格大模型(intent_model,空则回落主模型);降级路不实现运行时。
    llm = ChatOpenAI(
        model=settings.intent_model or settings.model,
        base_url=settings.intent_model_base_url or settings.base_url,
        api_key=settings.intent_model_api_key or settings.api_key,
        temperature=settings.temperature,
    )
    result = nodes.classify_intent(llm, query, _history_text(state))

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "route": route_from_intent(result.intent),
        "visited": ["intent"],
    }


def build_graph(checkpointer=None):
    """Build and compile the orchestration graph.

    Nodes reference ``nodes.*`` / ``agent.*`` via the module object so tests can
    monkeypatch them before building.
    """
    g = StateGraph(GraphState)
    g.add_node("reset", _reset_node)
    g.add_node("coref", _coref_node)
    g.add_node("intent", _intent_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("gate", nodes.gate_node)
    g.add_node("agent", agent.react_agent_node)
    g.add_node("fallback", nodes.fallback_node)
    g.add_node("complaint", nodes.complaint_node)
    g.add_node("script_reply", nodes.script_reply)
    g.add_node("fetch_order", nodes.fetch_order)
    g.add_node("retrieve_policy", nodes.retrieve_policy)
    g.add_node("log", nodes.log_node)

    g.add_edge(START, "reset")
    g.add_edge("reset", "coref")
    g.add_edge("coref", "intent")
    g.add_conditional_edges(
        "intent",
        lambda s: s["route"],
        {
            "escalate": "complaint",
            "fallback_script": "script_reply",
            "knowledge": "retrieve",
            "refund_flow": "fetch_order",
            "business": "agent",
        },
    )
    g.add_edge("retrieve", "gate")
    # 退款子流程确定性链:取单(缺单 interrupt 暂停)→ 检索政策 → 闸 → Agent
    g.add_edge("fetch_order", "retrieve_policy")
    g.add_edge("retrieve_policy", "gate")
    g.add_conditional_edges(
        "gate",
        lambda s: s.get("confidence_pass"),
        {True: "agent", False: "fallback"},
    )
    g.add_edge("agent", "log")
    g.add_edge("fallback", "log")
    g.add_edge("complaint", "log")
    g.add_edge("script_reply", "log")
    g.add_edge("log", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
