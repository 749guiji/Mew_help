"""Assemble the ch05 LangGraph orchestration graph."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import agent, nodes
from app.graph.router import route_from_intent
from app.graph.state import GraphState


def _coref_node(state: GraphState) -> dict:
    return {
        "resolved_query": nodes.resolve_coref(state["raw_query"]),
        "visited": ["coref"],
    }


def _intent_node(state: GraphState) -> dict:
    from app.core.config import get_settings
    from app.core.llm import get_llm

    intent = nodes.classify_intent(get_llm(get_settings()), state["resolved_query"])
    return {"intent": intent, "route": route_from_intent(intent), "visited": ["intent"]}


def build_graph(checkpointer=None):
    """Build and compile the orchestration graph.

    Nodes reference ``nodes.*`` / ``agent.*`` via the module object so tests can
    monkeypatch them before building.
    """
    g = StateGraph(GraphState)
    g.add_node("coref", _coref_node)
    g.add_node("intent", _intent_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("gate", nodes.gate_node)
    g.add_node("agent", agent.react_agent_node)
    g.add_node("fallback", nodes.fallback_node)
    g.add_node("complaint", nodes.complaint_node)
    g.add_node("chitchat", nodes.chitchat_node)
    g.add_node("log", nodes.log_node)

    g.add_edge(START, "coref")
    g.add_edge("coref", "intent")
    g.add_conditional_edges(
        "intent",
        lambda s: s["route"],
        {
            "knowledge": "retrieve",
            "business": "agent",
            "complaint": "complaint",
            "chitchat": "chitchat",
        },
    )
    g.add_edge("retrieve", "gate")
    g.add_conditional_edges(
        "gate", lambda s: s.get("confidence_pass"), {True: "agent", False: "fallback"}
    )
    g.add_edge("agent", "log")
    g.add_edge("fallback", "log")
    g.add_edge("complaint", "log")
    g.add_edge("chitchat", "log")
    g.add_edge("log", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())
