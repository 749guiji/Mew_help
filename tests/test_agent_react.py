"""Tests for the hand-rolled ReAct agent node (ch08: dynamic tools + pending_ticket)."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.agent import react_agent_node
from app.kb.hybrid import ScoredChunk
from app.tools.executor import ToolResult
from app.tools.registry import RegisteredTool


class _Dummy:
    def __init__(self, name):
        self.name = name

    async def ainvoke(self, args):
        return "dummy"


def _rt(name, read_only=True):
    return RegisteredTool(
        name=name,
        description="d",
        json_schema={"type": "object", "properties": {}},
        read_only=read_only,
        source="builtin",
        mcp_server=None,
        impl=_Dummy(name),
    )


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
        self.calls = []

    async def execute(self, tool, args, *, tool_call_id=None, conversation_id=None):
        self.calls.append((tool.name, args))
        return self.results.get(tool.name, ToolResult("ok", "ok"))


def _run(state, llm, executor, tools, max_steps=5, token_budget=None):
    return asyncio.run(
        react_agent_node(
            state, llm=llm, executor=executor, tools=tools, max_steps=max_steps,
            token_budget=token_budget,
        )
    )


def test_read_tool_call_returns_tool_message():
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        ),
        AIMessage(content="订单已发货"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({"query_order": ToolResult("ok", "订单 1001:已发货")})
    state = {"messages": [HumanMessage(content="订单 1001 到哪了")]}

    result = _run(state, llm, executor, [_rt("query_order")])

    assert any(
        isinstance(m, ToolMessage) and m.content == "订单 1001:已发货"
        for m in result["messages"]
    )
    assert result["messages"][-1].content == "订单已发货"


def test_failed_tool_records_error_status():
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        ),
        AIMessage(content="抱歉没查到"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({"query_order": ToolResult("error", "[工具错误] boom")})
    state = {"messages": [HumanMessage(content="订单 1001")]}

    result = _run(state, llm, executor, [_rt("query_order")])

    assert result["tool_status"]["c1"] == "error"


def test_multi_step():
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "query_product", "args": {"product_name": "猫粮"}, "id": "c2"}],
        ),
        AIMessage(content="订单在派送中"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor(
        {"query_order": ToolResult("ok", "已发货"), "query_product": ToolResult("ok", "有货")}
    )
    state = {"messages": [HumanMessage(content="订单 1001 的物流")]}

    result = _run(state, llm, executor, [_rt("query_order"), _rt("query_product")])

    assert llm.calls >= 3  # 至少三步(两次工具 + 一次收敛)
    assert result["messages"][-1].content == "订单在派送中"


def test_stop_on_max_steps():
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": f"c{i}"}],
        )
        for i in range(10)
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({"query_order": ToolResult("ok", "已发货")})
    state = {"messages": [HumanMessage(content="订单 1001")]}

    _run(state, llm, executor, [_rt("query_order")], max_steps=3)

    assert llm.calls == 3  # 达到 max_steps 停止,不无限循环


def test_missing_info_asks_user():
    llm = FakeLLM([AIMessage(content="请提供订单号")])
    executor = FakeExecutor()
    state = {"messages": [HumanMessage(content="帮我查订单")]}

    result = _run(state, llm, executor, [_rt("query_order")])

    assert result["messages"][-1].content == "请提供订单号"
    assert llm.calls == 1


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

    _run(state, llm, executor, [_rt("submit_refund")])

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

    _run(state, llm, executor, [_rt("query_order")])

    contents = [m.content for m in llm.last_messages if isinstance(m, HumanMessage)]
    assert not any("7天无理由" in c for c in contents)


def test_submit_refund_still_intercepted():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_refund",
                    "args": {"order_id": "1001", "reason": "质量问题"},
                    "id": "c1",
                }
            ],
        ),
        AIMessage(content="这一单可以退款。"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor({})
    state = {"messages": [HumanMessage(content="这单能退吗")]}

    result = _run(state, llm, executor, [_rt("submit_refund")])

    assert result["suggested_actions"][0]["type"] == "refund_form"
    assert executor.calls == []  # submit_refund 被拦截,不进引擎


def test_token_budget_trims_history():
    responses = [AIMessage(content="ok")]
    llm = FakeLLM(responses)
    executor = FakeExecutor()
    long_history = [HumanMessage(content="x" * 100) for _ in range(10)]
    state = {"messages": long_history}

    _run(state, llm, executor, [_rt("query_order")], token_budget=50)

    # 历史被截断:收到的消息数远少于 11(system + 10 条)
    assert len(llm.last_messages) <= 3


def test_create_ticket_valid_sets_pending_and_breaks():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {"description": "商品破损", "ticket_type": "售后"},
                    "id": "c1",
                }
            ],
        ),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor(
        {
            "create_ticket": ToolResult(
                "needs_confirmation",
                "",
                draft={"description": "商品破损", "ticket_type": "售后"},
            )
        }
    )
    state = {"messages": [HumanMessage(content="帮我建个工单")]}

    result = _run(state, llm, executor, [_rt("create_ticket", read_only=False)])

    assert result["pending_ticket"]["args"]["description"] == "商品破损"
    assert result["pending_ticket"]["tool_call_id"] == "c1"
    # 该轮不追加 create_ticket 的 ToolMessage(等确认后由 confirm_ticket 节点补)
    assert not any(
        isinstance(m, ToolMessage) and getattr(m, "tool_call_id", None) == "c1"
        for m in result["messages"]
    )


def test_create_ticket_validation_error_feeds_back():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "create_ticket", "args": {"ticket_type": "售后"}, "id": "c1"}
            ],
        ),
        AIMessage(content="请问您的问题描述是?"),
    ]
    llm = FakeLLM(responses)
    executor = FakeExecutor(
        {"create_ticket": ToolResult("validation_error", "参数校验失败:缺少 description")}
    )
    state = {"messages": [HumanMessage(content="帮我建个工单")]}

    result = _run(state, llm, executor, [_rt("create_ticket", read_only=False)])

    assert result["pending_ticket"] is None
    assert any(
        isinstance(m, ToolMessage) and "缺少 description" in m.content
        for m in result["messages"]
    )
    assert result["messages"][-1].content == "请问您的问题描述是?"
