"""Tests for the ch07 three-layer context assembly."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph.context import (
    assemble_history_context,
    assemble_model_context,
    split_layers,
    _truncate_layer2,
)


def test_truncate_layer2_user_untouched():
    assert _truncate_layer2(HumanMessage(content="它能退吗")) == "它能退吗"


def test_truncate_layer2_assistant_prefix():
    long = "这是一段很长的客服答复" * 10  # 100 chars
    out = _truncate_layer2(AIMessage(content=long))
    assert out.endswith("…") and len(out) == 41  # 40 字 + …
    assert _truncate_layer2(AIMessage(content="短答复")) == "短答复"


def test_truncate_layer2_tool_one_line():
    out = _truncate_layer2(
        ToolMessage(content="订单1001已发货", tool_call_id="c1", name="query_order")
    )
    assert out == "[工具:query_order]"


def test_assemble_model_context_order_and_layers():
    msgs = [
        HumanMessage("最开始的问题"),  # 层3
        AIMessage("最开始的回答"),
        HumanMessage("中间的问题"),  # 层2
        AIMessage("中间的回答" * 20),  # 层2,截短
        HumanMessage("最近的问题"),  # 层1
        AIMessage("最近的回答"),
        HumanMessage("当前这句话"),  # 当前轮
    ]
    out = assemble_model_context(
        msgs, summary_text="摘要内容", evidence="证据内容",
        summary_upto_pos=1, layer1_from_pos=4,
    )

    assert isinstance(out[0], SystemMessage)  # system 打头
    contents = [m.content for m in out]
    assert "中间的问题" in contents
    assert any(c.startswith("中间的回答") and c.endswith("…") for c in contents)  # 层2截短
    assert "最近的问题" in contents and "最近的回答" in contents  # 层1原文
    assert "当前这句话" in contents  # 当前轮
    assert not any("最开始的问题" == c for c in contents)  # 层3不渲染原文
    last = out[-1].content
    assert "摘要内容" in last and "证据内容" in last  # 摘要+证据合成一条挂最后


def test_split_layers_no_layer1_from():
    msgs = [HumanMessage("a"), AIMessage("b"), HumanMessage("c")]
    layer2, layer1, current = split_layers(msgs, summary_upto_pos=0, layer1_from_pos=None)
    assert layer2 == []  # 无层1起点 → 全进层1
    assert len(layer1) == 2  # a,b
    assert [m.content for m in current] == ["c"]


def test_assemble_history_context_summary_and_window():
    msgs = [
        HumanMessage("第一问"), AIMessage("第一答"),
        HumanMessage("第二问"), AIMessage("第二答"),
        HumanMessage("第三问"),
    ]
    out = assemble_history_context(msgs, summary_text="订单1001想退货", max_turns=2)
    assert "摘要:订单1001想退货" in out
    assert "第三问" in out
    assert "第一问" not in out  # 滑窗只留最近 2 轮
