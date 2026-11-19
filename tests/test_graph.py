"""End-to-end graph routing tests (ch05, with monkeypatched nodes)."""

import asyncio

from app.graph import agent, nodes
from app.graph.build import build_graph


def _invoke(graph, query):
    return asyncio.run(
        graph.ainvoke(
            {"messages": [("user", query)], "raw_query": query},
            config={"configurable": {"thread_id": "test-thread"}},
        )
    )


def test_knowledge_route_forces_retrieval(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q: "商品咨询")
    monkeypatch.setattr(
        nodes,
        "retrieve_node",
        lambda state: {"retrieval_chunks": [], "retrieval_score": 0.8, "visited": ["retrieve"]},
    )

    async def fake_agent(state):
        return {"messages": [("assistant", "答案")], "visited": ["agent"]}

    monkeypatch.setattr(agent, "react_agent_node", fake_agent)

    result = _invoke(build_graph(), "运费怎么算")
    assert "retrieve" in result["visited"]
    assert "gate" in result["visited"]
    assert "agent" in result["visited"]


def test_business_route_skips_retrieval(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q: "物流")

    async def fake_agent(state):
        return {"messages": [("assistant", "已查")], "visited": ["agent"]}

    monkeypatch.setattr(agent, "react_agent_node", fake_agent)

    result = _invoke(build_graph(), "订单 1001 物流")
    assert "retrieve" not in result["visited"]
    assert "agent" in result["visited"]


def test_complaint_returns_handoff_suggestions(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q: "投诉")

    result = _invoke(build_graph(), "我要投诉")

    assert result["handoff_suggestions"] == ["转人工", "建工单"]
    assert "agent" not in result["visited"]


def test_chitchat_fixed_reply(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q: "闲聊")

    result = _invoke(build_graph(), "你好")

    assert "agent" not in result["visited"]
    assert result["messages"][-1].content  # 固定话术非空


def test_knowledge_fallback_on_low_score(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q: "商品咨询")
    monkeypatch.setattr(
        nodes,
        "retrieve_node",
        lambda state: {"retrieval_chunks": [], "retrieval_score": 0.1, "visited": ["retrieve"]},
    )

    result = _invoke(build_graph(), "运费怎么算")

    assert "fallback" in result["visited"]
    assert "agent" not in result["visited"]
