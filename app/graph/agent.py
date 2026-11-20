"""Hand-rolled ReAct agent node (ch05).

A node that loops: invoke the tool-bound LLM → if there are tool calls, execute
them and feed the ToolMessages back → otherwise converge. Stop conditions are
``max_steps`` (no infinite loop) and an optional token budget.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.config import get_settings
from app.core.llm import get_llm
from app.prompts.system import SYSTEM_PROMPT
from app.tools import ALL_TOOLS
from app.tools.executor import ToolExecutor

# 主力 Agent 的工具 = ch02 五个去掉 query_faq(检索由知识节点做,避免双重检索)
# 和 create_ticket(建工单走前端 /api/tickets,不经过 Agent)。
AGENT_TOOLS = [t for t in ALL_TOOLS if t.name not in ("query_faq", "create_ticket")]


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

    history = [SystemMessage(content=SYSTEM_PROMPT), *state.get("messages", [])]
    if state.get("route") == "knowledge" and state.get("retrieval_chunks"):
        evidence = _format_evidence(state["retrieval_chunks"])
        history.append(
            HumanMessage(content=f"请仅依据以下知识片段回答用户问题:\n{evidence}")
        )
    if state.get("route") == "refund":
        parts = [
            "请判断这一单能否退款/退货,依据政策条款和订单状态作答,不得承诺退款金额或到账时间。"
        ]
        order = state.get("order_data")
        if order:
            parts.append(f"订单信息:{order}")
        if state.get("retrieval_chunks"):
            parts.append(f"政策条款:\n{_format_evidence(state['retrieval_chunks'])}")
        history.append(HumanMessage(content="\n\n".join(parts)))
    if len(history) == 1:  # 只有 system message,没有用户消息
        history.append(HumanMessage(content=state.get("resolved_query", "")))

    bound_llm = (llm or get_llm(get_settings())).bind_tools(
        tools if tools is not None else AGENT_TOOLS
    )
    exec_ = executor if executor is not None else ToolExecutor()

    produced: list = []
    for _ in range(max_steps):
        _trim_history(history, token_budget * 4)
        ai = await bound_llm.ainvoke(history)
        produced.append(ai)
        history.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            raw = await exec_.execute(tc["name"], tc["args"])
            tm = ToolMessage(content=raw, tool_call_id=tc["id"])
            produced.append(tm)
            history.append(tm)

    return {"messages": produced, "visited": ["agent"]}
