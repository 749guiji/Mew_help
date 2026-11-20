"""Slot/pause/resume tests for the ch06 refund sub-flow."""

import asyncio

import pytest

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


def _run_turns(graph, turns, thread="t"):
    config = {"configurable": {"thread_id": thread}}
    results = []
    for query, order_id in turns:
        payload = {"messages": [("user", query)], "raw_query": query}
        if order_id is not None:
            payload["order_id"] = order_id
        results.append(asyncio.run(graph.ainvoke(payload, config=config)))
    return results


def test_refund_without_order_id_pauses(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    result = _run_turns(build_graph(), [("这个能退吗", None)])[0]

    assert result["pending_slot"] == "order_id"
    assert len(result["order_list"]) >= 2
    assert "请选择" in result["messages"][-1].content
    assert "agent" not in result["visited"]


def test_refund_pause_then_resume(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    r1, r2 = _run_turns(build_graph(), [("这个能退吗", None), ("已选择订单", "1001")])

    assert r1["pending_slot"] == "order_id"
    assert r2["expanded_queries"] == ["退款政策", "退货条件"]
    assert len(r2["retrieval_chunks"]) == 1
    assert r2["order_data"]["order_id"] == "1001"
    assert "agent" in r2["visited"]
    assert r2["order_id"] is None
    assert r2["pending_slot"] is None


def test_refund_abandon_clears_order_id(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: [])
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    graph = build_graph()
    r1 = _run_turns(graph, [("这个能退吗", None)])[0]
    assert r1["pending_slot"] == "order_id"

    monkeypatch.setattr(
        nodes, "classify_intent", lambda llm, q, h="": IntentResult("物流", 0.9)
    )
    r2 = _run_turns(graph, [("物流到哪了", None)])[0]

    assert r2["order_id"] is None
    assert r2["pending_slot"] is None
    assert r2["route"] == "business"
    assert "agent" in r2["visited"]


def test_refund_second_time_reprompts(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    r1, r2, r3 = _run_turns(
        build_graph(),
        [("这个能退吗", None), ("已选择订单", "1001"), ("这个还能退吗", None)],
    )

    assert r1["pending_slot"] == "order_id"
    assert r2["order_id"] is None
    assert r3["pending_slot"] == "order_id"
    assert r3["order_id"] is None
    assert len(r3["order_list"]) >= 2


def test_refund_zero_chunks_falls_back(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: [])
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    r1, r2 = _run_turns(build_graph(), [("这个能退吗", None), ("已选择订单", "1001")])

    assert "fallback" in r2["visited"]
    assert "agent" not in r2["visited"]


def test_order_id_without_pending_subflow_does_not_crash(monkeypatch):
    import app.kb.retrieval as retrieval_mod

    monkeypatch.setattr(retrieval_mod, "retrieve_multi", lambda queries, **kw: _chunks())
    monkeypatch.setattr(agent, "react_agent_node", _fake_agent)

    graph = build_graph()
    payload = {"messages": [("user", "查订单")], "raw_query": "查订单", "order_id": "1001"}
    result = asyncio.run(
        graph.ainvoke(payload, config={"configurable": {"thread_id": "fresh"}})
    )

    # 无挂起子流程却带 order_id:当作普通新问题走 coref/intent,不崩
    assert "coref" in result["visited"]
    assert "intent" in result["visited"]
