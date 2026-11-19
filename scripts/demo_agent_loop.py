"""最裸的 Agent 循环(不用 LangGraph)。

演示「Agent 就是带工具的循环」:
1. 调 LLM(绑工具)
2. 有 tool_call → 执行工具 → 结果喂回 → 回到 1
3. 无 tool_call → 收敛,打印答案
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage, ToolMessage

from app.core.config import get_settings
from app.core.llm import get_llm
from app.tools import TOOL_REGISTRY
from app.tools.executor import ToolExecutor

# 只挂两个业务工具做演示
DEMO_TOOLS = [TOOL_REGISTRY["query_order"], TOOL_REGISTRY["query_logistics"]]


async def run(query: str) -> None:
    llm = get_llm(get_settings()).bind_tools(DEMO_TOOLS)
    executor = ToolExecutor()
    messages = [HumanMessage(content=query)]

    for step in range(1, 6):  # max 5 步
        ai = await llm.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"[收敛] {ai.content}")
            return
        for tc in ai.tool_calls:
            print(f"[工具调用] {tc['name']}({tc['args']})")
            raw = await executor.execute(tc["name"], tc["args"])
            print(f"[工具结果] {raw}")
            messages.append(ToolMessage(content=raw, tool_call_id=tc["id"]))
    print("[停止] 达到最大步数")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "订单 1001 的物流到哪了"
    asyncio.run(run(query))
