"""MCP Client:用 langchain-mcp-adapters 的 MultiServerMCPClient 现问现拿工具。"""

from __future__ import annotations

import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import get_settings
from app.tools.registry import RegisteredTool, json_schema_from_tool, registry

logger = logging.getLogger(__name__)

# (server 名, 配置字段名)
_SERVERS = [
    ("logistics", "mcp_logistics_url"),
    ("aftersale", "mcp_aftersale_url"),
]

_client: MultiServerMCPClient | None = None


def _get_client() -> MultiServerMCPClient:
    """惰性建单例(只持连接配置,不含会话态)。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = MultiServerMCPClient(
            {
                name: {"transport": "streamable_http", "url": getattr(settings, field)}
                for name, field in _SERVERS
            }
        )
    return _client


async def discover_mcp_tools(client=None) -> list[RegisteredTool]:
    """每轮现问现拿:对每个已配置 server 列工具并包成 RegisteredTool。

    - 连不上的 server 记 warn 跳过、不缓存旧工具(宕机降级);
    - 与内置同名的 MCP 工具跳过(内置优先)。
    """
    client = client if client is not None else _get_client()
    settings = get_settings()
    write_allowlist = set(settings.mcp_write_tools)
    builtin_names = {t.name for t in registry.builtin()}

    out: list[RegisteredTool] = []
    for name, _ in _SERVERS:
        try:
            server_tools = await client.get_tools(server_name=name)
        except Exception as e:  # noqa: BLE001 - 宕机降级,不崩整轮
            logger.warning("MCP server %s unreachable, skipping: %s", name, e)
            continue
        for t in server_tools:
            if t.name in builtin_names:
                logger.warning("MCP tool %s.%s collides with builtin, skipping", name, t.name)
                continue
            out.append(
                RegisteredTool(
                    name=t.name,
                    description=t.description,
                    json_schema=json_schema_from_tool(t),
                    read_only=t.name not in write_allowlist,
                    source="mcp",
                    mcp_server=name,
                    impl=t,
                )
            )
    return out
