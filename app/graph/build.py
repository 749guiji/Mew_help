"""Assemble the ch06 LangGraph orchestration graph."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import agent, nodes
from app.graph.router import route_from_intent
from app.graph.state import GraphState


def _reset_node(state: GraphState) -> dict:
    """清空上一轮的轮内字段;不碰子流程上下文(resolved_query/route/intent/order_id/pending_slot)。"""
    return {
        "retrieval_chunks": [],
        "retrieval_score": None,
        "confidence_pass": None,
        "handoff_suggestions": [],
        "expanded_queries": [],
        "order_data": None,
        "order_list": [],
        "intent_confidence": None,
    }


def _slot_node(state: GraphState) -> dict:
    """订单槽位:带 order_id = 恢复挂起的子流程;无 = 新问题(弃选则清挂起上下文)。"""
    if state.get("order_id"):
        # 恢复:保留 resolved_query/route/intent,清 pending_slot,跳 get_order
        return {"pending_slot": None, "visited": ["slot"]}
    # 新问题 / 弃选:清掉可能挂起的子流程上下文
    return {
        "order_id": None,
        "pending_slot": None,
        "route": "",
        "resolved_query": "",
        "intent": "",
        "intent_confidence": None,
        "visited": ["slot"],
    }


def _coref_node(state: GraphState) -> dict:
    from app.core.config import get_settings
    from app.core.llm import get_llm

    history = [m.content for m in state.get("messages", []) if getattr(m, "content", None)]
    resolved = nodes.resolve_coref(get_llm(get_settings()), state["raw_query"], history)
    return {"resolved_query": resolved, "visited": ["coref"]}


def _intent_node(state: GraphState) -> dict:
    from langchain_openai import ChatOpenAI

    from app.core.config import get_settings

    settings = get_settings()
    query = state["resolved_query"]

    big_llm = ChatOpenAI(
        model=settings.intent_model,
        base_url=settings.intent_model_base_url,
        api_key=settings.intent_model_api_key,
        temperature=settings.temperature,
    )

    if settings.intent_small_model and settings.intent_small_model != settings.intent_model:
        small_llm = ChatOpenAI(
            model=settings.intent_small_model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            temperature=settings.temperature,
        )
        result = nodes.classify_intent(small_llm, query)
        result = nodes.maybe_escalate(
            big_llm, query, result, settings.intent_escalate_threshold
        )
    else:
        result = nodes.classify_intent(big_llm, query)

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "route": route_from_intent(result.intent),
        "visited": ["intent"],
    }


def _get_order_node(state: GraphState) -> dict:
    from app.orders import get_order, list_orders

    if not state.get("order_id"):
        return {
            "order_list": list_orders(),
            "pending_slot": "order_id",
            "messages": [AIMessage(content="请选择要咨询的订单：")],
            "visited": ["get_order"],
        }
    return {"order_data": get_order(state["order_id"]), "visited": ["get_order"]}


def _expand_node(state: GraphState) -> dict:
    from app.core.config import get_settings
    from app.core.llm import get_llm

    queries = nodes.expand_queries(get_llm(get_settings()), state["resolved_query"])
    return {"expanded_queries": queries, "visited": ["expand"]}


def _retrieve_policy_node(state: GraphState) -> dict:
    from app.kb.retrieval import retrieve_multi

    queries = state.get("expanded_queries") or [state["resolved_query"]]
    chunks = retrieve_multi(queries)
    score = max((c.score for c in chunks), default=None) if chunks else None
    return {"retrieval_chunks": chunks, "retrieval_score": score, "visited": ["retrieve_policy"]}


async def _agent_node(state: GraphState) -> dict:
    result = dict(await agent.react_agent_node(state))
    if state.get("route") == "refund":
        # 退款子流程出最终答案后清槽位,同会话下次退款提问重新弹选择器
        result["order_id"] = None
        result["pending_slot"] = None
    return result


def build_graph(checkpointer=None):
    """Build and compile the orchestration graph.

    Nodes reference ``nodes.*`` / ``agent.*`` via the module object so tests can
    monkeypatch them before building.
    """
    g = StateGraph(GraphState)
    g.add_node("reset", _reset_node)
    g.add_node("slot", _slot_node)
    g.add_node("coref", _coref_node)
    g.add_node("intent", _intent_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("gate", nodes.gate_node)
    g.add_node("agent", _agent_node)
    g.add_node("fallback", nodes.fallback_node)
    g.add_node("complaint", nodes.complaint_node)
    g.add_node("chitchat", nodes.chitchat_node)
    g.add_node("get_order", _get_order_node)
    g.add_node("expand", _expand_node)
    g.add_node("retrieve_policy", _retrieve_policy_node)
    g.add_node("log", nodes.log_node)

    g.add_edge(START, "reset")
    g.add_edge("reset", "slot")
    g.add_conditional_edges(
        "slot",
        lambda s: "get_order" if s.get("order_id") else "coref",
        {"get_order": "get_order", "coref": "coref"},
    )
    g.add_edge("coref", "intent")
    g.add_conditional_edges(
        "intent",
        lambda s: s["route"],
        {
            "knowledge": "retrieve",
            "refund": "get_order",
            "business": "agent",
            "complaint": "complaint",
            "chitchat": "chitchat",
        },
    )
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges(
        "get_order",
        lambda s: "expand" if s.get("order_id") else "log",
        {"expand": "expand", "log": "log"},
    )
    g.add_edge("expand", "retrieve_policy")
    g.add_edge("retrieve_policy", "gate")
    g.add_conditional_edges(
        "gate",
        lambda s: s.get("confidence_pass"),
        {True: "agent", False: "fallback"},
    )
    g.add_edge("agent", "log")
    g.add_edge("fallback", "log")
    g.add_edge("complaint", "log")
    g.add_edge("chitchat", "log")
    g.add_edge("log", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
