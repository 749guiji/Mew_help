"""End-to-end graph routing tests (ch05, with monkeypatched nodes)."""

import asyncio

import pytest

from app.graph import agent, nodes
from app.graph.build import build_graph
from app.graph.nodes import IntentResult


@pytest.fixture(autouse=True)
def _no_real_coref(monkeypatch):
    """ch06 起 resolve_coref 走 LLM;图测试只关心路由,原样透传避开真模型。"""
    monkeypatch.setattr(nodes, "resolve_coref", lambda llm, q, h: q)


def _invoke(graph, query):
    return asyncio.run(
        graph.ainvoke(
            {"messages": [("user", query)], "raw_query": query},
            config={"configurable": {"thread_id": "test-thread"}},
        )
    )


def test_knowledge_route_forces_retrieval(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("商品咨询", 0.9))
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
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("物流", 0.9))

    async def fake_agent(state):
        return {"messages": [("assistant", "已查")], "visited": ["agent"]}

    monkeypatch.setattr(agent, "react_agent_node", fake_agent)

    result = _invoke(build_graph(), "订单 1001 物流")
    assert "retrieve" not in result["visited"]
    assert "agent" in result["visited"]


def test_complaint_returns_handoff_suggestions(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("投诉", 0.9))

    result = _invoke(build_graph(), "我要投诉")

    assert result["handoff_suggestions"] == ["转人工", "建工单"]
    assert "agent" not in result["visited"]


def test_fallback_script_fixed_reply(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("闲聊", 0.9))

    result = _invoke(build_graph(), "你好")

    assert "agent" not in result["visited"]
    assert "script_reply" in result["visited"]
    assert result["messages"][-1].content  # 固定话术非空


def test_other_intent_fallback_script(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("其他", 0.2))

    result = _invoke(build_graph(), "帮我写首诗")

    assert "agent" not in result["visited"]
    assert "script_reply" in result["visited"]
    assert "具体" in result["messages"][-1].content  # 其他类请说具体些


def test_knowledge_fallback_on_low_score(monkeypatch):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("商品咨询", 0.9))
    monkeypatch.setattr(
        nodes,
        "retrieve_node",
        lambda state: {"retrieval_chunks": [], "retrieval_score": 0.1, "visited": ["retrieve"]},
    )

    result = _invoke(build_graph(), "运费怎么算")

    assert "fallback" in result["visited"]
    assert "agent" not in result["visited"]


def test_two_turns_reset_per_turn_fields(monkeypatch):
    graph = build_graph()
    config = {"configurable": {"thread_id": "two-turn"}}

    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("投诉", 0.9))
    r1 = asyncio.run(
        graph.ainvoke(
            {"messages": [("user", "我要投诉")], "raw_query": "我要投诉"},
            config=config,
        )
    )
    assert r1["handoff_suggestions"] == ["转人工", "建工单"]

    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("闲聊", 0.9))
    r2 = asyncio.run(
        graph.ainvoke(
            {"messages": [("user", "你好")], "raw_query": "你好"},
            config=config,
        )
    )

    assert r2.get("handoff_suggestions") == []  # reset node 清掉了上一轮的 handoff
