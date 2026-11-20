"""Interrupt/resume tests for the ch06 refund sub-flow (fetch_order)."""

import asyncio

import pytest
from langgraph.types import Command

from app.graph import agent, nodes
from app.graph.build import build_graph
from app.graph.nodes import IntentResult
from app.kb.hybrid import ScoredChunk


@pytest.fixture(autouse=True)
def _fake_understanding(monkeypatch):
    """避开真模型:coref 透传、意图固定退款退货、扩写固定两条。"""
    monkeypatch.setattr(nodes, "resolve_coref", lambda llm, q, h: q)
    monkeypatch.setattr(
        nodes, "classify_intent", lambda llm, q, h="": IntentResult("退款退货", 0.9)
    )
    monkeypatch.setattr(
        nodes, "expand_queries", lambda llm, q: ["退款政策", "退货条件"]
    )


async def _fake_agent(state):
    return {"messages": [("assistant", "这单可以退")], "visited": ["agent"]}


def _chunks():
    return [ScoredChunk(1, 0.9, "退款政策", "退款", "policy", "7天无理由退货", "")]


def _invoke(graph, query, thread):
    config = {"configurable": {"thread_id": thread}}
    return asyncio.run(
        graph.ainvoke({"messages": [("user", query)], "raw_query": query}, config)
    )


def _resume(graph, order_id, thread):
    config = {"configurable": {"thread_id": thread}}
    return asyncio.run(graph.ainvoke(Command(resume=order_id), config))


def test_refund_without_order_id_interrupts(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    result = _invoke(build_graph(), "我要退款", "t-interrupt")

    intr = result.get("__interrupt__")
    assert intr and intr[0].value["type"] == "select_order"
    assert len(intr[0].value["orders"]) >= 2
    assert "agent" not in result.get("visited", [])


def test_refund_interrupt_then_resume(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    graph = build_graph()
    r1 = _invoke(graph, "我要退款", "t-resume")
    order_id = r1["__interrupt__"][0].value["orders"][0]["order_id"]

    r2 = _resume(graph, order_id, "t-resume")

    assert "fetch_order" in r2["visited"]
    assert "retrieve_policy" in r2["visited"]
    assert "agent" in r2["visited"]
    assert r2["order_data"]["order_id"] == order_id
    assert len(r2["retrieval_chunks"]) == 1


def test_refund_with_order_id_in_query_no_interrupt(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    result = _invoke(build_graph(), "订单1001能退吗", "t-with-id")

    assert "__interrupt__" not in result
    assert "fetch_order" in result["visited"]
    assert result["order_id"] == "1001"
    assert "agent" in result["visited"]


def test_refund_zero_chunks_falls_back(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: [])
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    result = _invoke(build_graph(), "订单1001能退吗", "t-fallback")

    assert "fallback" in result["visited"]
    assert "agent" not in result["visited"]


def test_refund_second_time_reinterrupts(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    graph = build_graph()
    r1 = _invoke(graph, "我要退款", "t-second")
    oid = r1["__interrupt__"][0].value["orders"][0]["order_id"]
    _resume(graph, oid, "t-second")

    r3 = _invoke(graph, "再退另一单", "t-second")
    assert r3.get("__interrupt__")  # 重新弹选择器,不复用旧 order_id
