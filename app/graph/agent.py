"""Hand-rolled ReAct agent node (ch05).

A node that loops: invoke the tool-bound LLM → if there are tool calls, execute
them and feed the ToolMessages back → otherwise converge. Stop conditions are
``max_steps`` (no infinite loop) and an optional token budget.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.core.config import get_settings
from app.core.llm import get_llm
from app.tools import ALL_TOOLS
from app.tools.executor import ToolExecutor

# 主力 Agent 的工具 = ch02 五个去掉 query_faq(检索由知识节点做,避免双重检索)
AGENT_TOOLS = [t for t in ALL_TOOLS if t.name != "query_faq"]


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
    history = list(state.get("messages", []))
    if state.get("route") == "knowledge" and state.get("retrieval_chunks"):
        evidence = _format_evidence(state["retrieval_chunks"])
        history = history + [
            HumanMessage(content=f"请仅依据以下知识片段回答用户问题:\n{evidence}")
        ]
    if not history:
        history = [HumanMessage(content=state.get("resolved_query", ""))]

    bound_llm = (llm or get_llm(get_settings())).bind_tools(
        tools if tools is not None else AGENT_TOOLS
    )
    exec_ = executor if executor is not None else ToolExecutor()

    produced: list = []
    for _ in range(max_steps):
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

        if token_budget is not None:
            chars = sum(len(m.content or "") for m in history)
            if chars > token_budget * 4:  # 粗略:中文约 1 token ≈ 1 字,留 4 倍余量
                break

    return {"messages": produced, "visited": ["agent"]}
