"""Tool executor with timeout and retry; never raises."""

import asyncio

from app.tools import TOOL_REGISTRY


class ToolExecutor:
    def __init__(self, registry=None, timeout: float = 5.0, retries: int = 2):
        self.registry = registry if registry is not None else TOOL_REGISTRY
        self.timeout = timeout
        self.retries = retries

    async def execute(self, name: str, args: dict) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return f"[工具错误] 未找到工具: {name}"

        last_exc: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                result = await asyncio.wait_for(tool.ainvoke(args), self.timeout)
                return str(result)
            except Exception as exc:
                last_exc = exc
        return f"[工具错误] {last_exc}"
