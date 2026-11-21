"""Smoke tests for the ch08 business MCP servers (logistics / aftersale)."""

import asyncio

from mcp_servers import aftersale_server, logistics_server


def _tool_names(mcp):
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_logistics_has_query_logistics():
    assert logistics_server.mcp.name == "logistics"
    assert "query_logistics" in _tool_names(logistics_server.mcp)


def test_aftersale_has_two_tools():
    assert aftersale_server.mcp.name == "aftersale"
    assert _tool_names(aftersale_server.mcp) == {"check_warranty", "check_return_progress"}


def test_mock_tools_return_str():
    assert isinstance(logistics_server.query_logistics("1001"), str)
    assert isinstance(aftersale_server.check_warranty("1001"), str)
    assert isinstance(aftersale_server.check_return_progress("1001"), str)
