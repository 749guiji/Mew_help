"""Tests for the hand-rolled ReAct agent node (app.graph.agent)."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.agent import AGENT_TOOLS, react_agent_node


class FakeLLM:
    """bind_tools returns self; ainvoke pops the next canned response."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
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


def test_agent_tools_exclude_query_faq():
    assert all(t.name != "query_faq" for t in AGENT_TOOLS)
