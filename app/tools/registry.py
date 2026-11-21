"""Pluggable tool registry (ch08).

``RegisteredTool`` 是「工具名 / 用途描述 / JSON Schema / 读写标记 / 来源 / 实现」
的统一登记条目;``ToolRegistry`` 存同一份表。内置工具启动时 ``register``,MCP
工具每轮 ``discover`` 现问现拿;主力 Agent 每轮从 ``registry`` 取并集。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RegisteredTool:
    name: str
    description: str
    json_schema: dict
    read_only: bool
    source: str  # "builtin" | "mcp"
    mcp_server: str | None
    impl: Any  # 有 .ainvoke(args) 的可调用


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def builtin(self) -> list[RegisteredTool]:
        return [t for t in self._tools.values() if t.source == "builtin"]

    def agent_builtin(self) -> list[RegisteredTool]:
        """主力 Agent 的内置工具集:全部内置去掉 query_faq(检索由知识节点做)。"""
        return [t for t in self.builtin() if t.name != "query_faq"]


def json_schema_from_tool(tool) -> dict:
    """取一个 langchain 工具的 JSON Schema 参数定义。

    内置 ``@tool`` 的 ``args_schema`` 是 pydantic 模型(MCP 工具可能是 dict),
    这里统一归一成 JSON Schema dict。
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return {"type": "object", "properties": {}}
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema if isinstance(schema, dict) else {"type": "object", "properties": {}}


registry = ToolRegistry()
