"""Tool layer: pluggable registry (ch08).

内置工具在这里启动时登记进 ``registry``;MCP 工具由 ``app.mcp.client`` 每轮
``discover`` 后与内置合成一份。``query_logistics`` 内置已下线(由物流 MCP
Server 接管)。
"""

from app.tools.context import current_conversation_id
from app.tools.registry import RegisteredTool, json_schema_from_tool, registry

from app.tools.business import query_order, query_product, submit_refund
from app.tools.db_tools import create_ticket, query_faq


def _register_builtin(name: str, tool, *, read_only: bool = True) -> None:
    registry.register(
        RegisteredTool(
            name=name,
            description=tool.description,
            json_schema=json_schema_from_tool(tool),
            read_only=read_only,
            source="builtin",
            mcp_server=None,
            impl=tool,
        )
    )


_register_builtin("query_order", query_order)
_register_builtin("query_product", query_product)
_register_builtin("submit_refund", submit_refund)
_register_builtin("query_faq", query_faq)
_register_builtin("create_ticket", create_ticket, read_only=False)

__all__ = [
    "registry",
    "RegisteredTool",
    "json_schema_from_tool",
    "current_conversation_id",
]
