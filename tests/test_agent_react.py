"""Tests for the hand-rolled ReAct agent node (app.graph.agent)."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.agent import AGENT_TOOLS, react_agent_node
from app.kb.hybrid import ScoredChunk


class FakeLLM:
    """bind_tools returns self; ainvoke pops the next canned response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_messages = list(messages)
        if self.responses:
            return self.responses.pop(0)
        return AIMessage(content="")


class FakeExecutor:
    def __init__(self, results=None):
        self.results = results or {}

    async def execute(self, name, args):
        return self.results.get(name, "ok")


def _run(state, llm, executor, max_steps=5):
    return asyncio.run(
        react_agent_node(state, llm=llm, executor=executor, max_steps=max_steps)
    )


def test_single_tool_call_converges():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="订单已发货"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({"query_order": "订单 1001:已发货"})
    state = {"messages": [HumanMessage(content="订单 1001 到哪了")]}

    result = _run(state, llm, executor)

    assert any(isinstance(m, ToolMessage) for m in result["messages"])
    assert result["messages"][-1].content == "订单已发货"


def test_multi_step():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2"}]),
        AIMessage(content="订单在派送中"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({"query_order": "已发货", "query_logistics": "派送中"})
    state = {"messages": [HumanMessage(content="订单 1001 的物流")]}

    result = _run(state, llm, executor)

    assert llm.calls >= 3  # 至少三步(两次工具 + 一次收敛)
    assert result["messages"][-1].content == "订单在派送中"


def test_stop_on_max_steps():
    responses = [
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": f"c{i}"}])
        for i in range(10)
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({"query_order": "已发货"})
    state = {"messages": [HumanMessage(content="订单 1001")]}

    _run(state, llm, executor, max_steps=3)

    assert llm.calls == 3  # 达到 max_steps 停止,不无限循环


def test_missing_info_asks_user():
    llm = FakeLLM([AIMessage(content="请提供订单号")])
    executor = FakeExecutor()
    state = {"messages": [HumanMessage(content="帮我查订单")]}

    result = _run(state, llm, executor)

    assert result["messages"][-1].content == "请提供订单号"
    assert llm.calls == 1


def test_agent_tools_exclude_query_faq_and_create_ticket():
    assert all(t.name not in ("query_faq", "create_ticket") for t in AGENT_TOOLS)


def test_refund_route_injects_order_and_evidence():
    llm = FakeLLM([AIMessage(content="这单可以退")])
    executor = FakeExecutor()
    state = {
        "messages": [HumanMessage(content="这个能退吗")],
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "product": "耳机", "status": "已签收"},
        "retrieval_chunks": [
            ScoredChunk(1, 0.9, "退款政策", "退款", "policy", "7天无理由退货", "")
        ],
    }

    _run(state, llm, executor)

    contents = [m.content for m in llm.last_messages if isinstance(m, HumanMessage)]
    assert any("1001" in c for c in contents)  # 注入了订单号
    assert any("7天无理由" in c for c in contents)  # 注入了政策证据


def test_business_route_does_not_inject_refund_evidence():
    llm = FakeLLM([AIMessage(content="已查")])
    executor = FakeExecutor()
    state = {
        "messages": [HumanMessage(content="物流到哪了")],
        "route": "business",
        "order_data": {"order_id": "1001"},
        "retrieval_chunks": [
            ScoredChunk(1, 0.9, "退款政策", "退款", "policy", "7天无理由退货", "")
        ],
    }

    _run(state, llm, executor)

    contents = [m.content for m in llm.last_messages if isinstance(m, HumanMessage)]
    assert not any("7天无理由" in c for c in contents)


def test_submit_refund_intercepted_as_refund_form():
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "submit_refund", "args": {"order_id": "1001"}, "id": "r1"}],
        ),
        AIMessage(content="这单可以退"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor()
    state = {"messages": [HumanMessage(content="这单能退吗")], "route": "refund_flow"}

    result = _run(state, llm, executor)

    assert result["suggested_actions"] == [
        {"type": "refund_form", "draft": {"order_id": "1001", "reason": None}}
    ]
    tms = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.name == "submit_refund" and "提交退款工单" in m.content for m in tms)


def test_token_budget_trims_history():
    responses = [AIMessage(content="ok")]
    llm = FakeLLM(responses)
    executor = FakeExecutor()
    long_history = [HumanMessage(content="x" * 100) for _ in range(10)]
    state = {"messages": long_history}

    asyncio.run(react_agent_node(state, llm=llm, executor=executor, token_budget=50))

    # 历史被截断:收到的消息数远少于 11(system + 10 条)
    assert len(llm.last_messages) <= 3
