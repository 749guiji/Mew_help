"""Hand-rolled ReAct agent node (ch05; ch08: 动态取工具 + pending_ticket).

A node that loops: invoke the tool-bound LLM → if there are tool calls, execute
them and feed the ToolMessages back → otherwise converge. Stop conditions are
``max_steps`` (no infinite loop) and an optional token budget.

ch08:工具来自注册中心(内置 + MCP 现问现拿);写工具(create_ticket)经引擎返回
``needs_confirmation`` 时记 ``pending_ticket`` 并停止,交给 ``confirm_ticket``
节点走 interrupt 预览确认。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.config import get_settings
from app.core.llm import get_llm
from app.tools.executor import ToolExecutor
from app.tools.registry import registry


def _trim_history(history: list, budget_chars: int) -> list:
    """Drop oldest non-system messages until the history fits ``budget_chars``."""
    while len(history) > 2:
        total = sum(len(m.content or "") for m in history)
        if total <= budget_chars:
            break
        history.pop(1)  # keep index 0 (system message)
    return history


def _format_evidence(chunks) -> str:
    """知识类:把检索块格式化成带 [n] 编号的证据文本(复用 ch04)。"""
    from app.services.generation_quality import format_evidence

    return format_evidence(chunks)


async def _resolve_tools(tools):
    """tools=None 时动态取:内置 + MCP 现问现拿(新工具即用、不重启)。"""
    if tools is not None:
        return tools
    from app.mcp.client import discover_mcp_tools

    return registry.agent_builtin() + await discover_mcp_tools()


async def react_agent_node(
    state: dict,
    *,
    llm=None,
    tools=None,
    executor=None,
    max_steps: int = 5,
    token_budget: int | None = None,
) -> dict:
    """Run the ReAct loop and return the new AI/Tool messages for the reducer."""
    if token_budget is None:
        token_budget = get_settings().max_history_tokens

    from app.graph.context import assemble_model_context, split_layers
    from app.graph.context_meta import load_context_meta

    summary_text, summary_upto_pos, layer1_from_pos = load_context_meta(
        state.get("conversation_id")
    )

    # 降级接线:层1超预算上移锚点、层2超预算后台起摘要
    from app.core.degradation import apply_degradation

    _l2, _l1, _ = split_layers(
        state.get("messages", []),
        summary_upto_pos=summary_upto_pos,
        layer1_from_pos=layer1_from_pos,
    )
    apply_degradation(
        state.get("conversation_id"),
        sum(len(m.content or "") for m in _l1),
        sum(len(m.content or "") for m in _l2),
    )

    evidence = ""
    if state.get("route") == "knowledge" and state.get("retrieval_chunks"):
        evidence = f"请仅依据以下知识片段回答用户问题:\n{_format_evidence(state['retrieval_chunks'])}"
    if state.get("route") == "refund_flow":
        parts = [
            "请判断这一单能否退款/退货,依据政策条款和订单状态作答,不得承诺退款金额或到账时间。"
            "若能退,调用 submit_refund 工具(传该订单号);若不能退,不要调用任何工具,说清原因并引导人工客服。"
        ]
        order = state.get("order_data")
        if order:
            parts.append(f"订单信息:{order}")
        if state.get("retrieval_chunks"):
            parts.append(f"政策条款:\n{_format_evidence(state['retrieval_chunks'])}")
        evidence = "\n\n".join(parts)

    history = assemble_model_context(
        state.get("messages", []),
        summary_text=summary_text,
        evidence=evidence,
        summary_upto_pos=summary_upto_pos,
        layer1_from_pos=layer1_from_pos,
    )
    if len(history) == 1:  # 只有 system message,没有用户消息
        history.append(HumanMessage(content=state.get("resolved_query", "")))

    toolset = await _resolve_tools(tools)
    name_to_tool = {t.name: t for t in toolset}
    bound_llm = (llm or get_llm(get_settings())).bind_tools([t.impl for t in toolset])
    exec_ = executor if executor is not None else ToolExecutor()
    conversation_id = state.get("conversation_id")

    produced: list = []
    suggested_actions: list = []
    pending_ticket: dict | None = None
    for _ in range(max_steps):
        _trim_history(history, token_budget * 4)
        ai = await bound_llm.ainvoke(history)
        produced.append(ai)
        history.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            if tc["name"] == "submit_refund":
                # 拦截:不执行,合成 ToolMessage 促收敛,记 refund_form 前端动作
                suggested_actions.append(
                    {
                        "type": "refund_form",
                        "draft": {
                            "order_id": tc["args"].get("order_id", ""),
                            "reason": tc["args"].get("reason"),
                        },
                    }
                )
                tm = ToolMessage(
                    content="已把「提交退款工单」选项交给用户确认。请用一句话说明这一单可以退款并停止,不要再调用任何工具。",
                    tool_call_id=tc["id"],
                    name="submit_refund",
                )
            else:
                rt = name_to_tool.get(tc["name"])
                if rt is None:
                    tm = ToolMessage(
                        content="[工具错误] 未找到工具", tool_call_id=tc["id"], name=tc["name"]
                    )
                else:
                    result = await exec_.execute(
                        rt, tc["args"], tool_call_id=tc["id"], conversation_id=conversation_id
                    )
                    if result.kind == "needs_confirmation":
                        pending_ticket = {"tool_call_id": tc["id"], "args": result.draft}
                        break  # 交 confirm_ticket 节点,不再执行
                    tm = ToolMessage(content=result.content, tool_call_id=tc["id"], name=tc["name"])
            produced.append(tm)
            history.append(tm)
        if pending_ticket is not None:
            break

    return {
        "messages": produced,
        "suggested_actions": suggested_actions,
        "pending_ticket": pending_ticket,
        "visited": ["agent"],
    }
