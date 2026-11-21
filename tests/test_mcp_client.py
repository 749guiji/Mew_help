"""Tests for the MCP client discovery (ch08)."""

import asyncio

from app.mcp.client import discover_mcp_tools


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = "d"
        self.args_schema = {"type": "object", "properties": {}}


class FakeClient:
    def __init__(self, tools_by_server, fail=()):
        self.tools_by_server = tools_by_server
        self.fail = set(fail)

    async def get_tools(self, server_name=None):
        if server_name in self.fail:
            raise RuntimeError("unreachable")
        return self.tools_by_server.get(server_name, [])


def test_discover_tags_source_and_server():
    client = FakeClient(
        {
            "logistics": [FakeTool("query_logistics")],
            "aftersale": [FakeTool("check_warranty"), FakeTool("check_return_progress")],
        }
    )
    tools = asyncio.run(discover_mcp_tools(client=client))
    assert {t.source for t in tools} == {"mcp"}
    assert {t.mcp_server for t in tools} == {"logistics", "aftersale"}
    assert {t.name for t in tools} == {"query_logistics", "check_warranty", "check_return_progress"}


def test_discover_skips_unreachable_server():
    client = FakeClient({"logistics": [FakeTool("query_logistics")]}, fail={"aftersale"})
    tools = asyncio.run(discover_mcp_tools(client=client))
    assert {t.mcp_server for t in tools} == {"logistics"}  # aftersale 跳过,不抛


def test_discover_skips_builtin_name_collision():
    client = FakeClient({"logistics": [FakeTool("query_order")]})  # query_order 是内置工具
    tools = asyncio.run(discover_mcp_tools(client=client))
    assert "query_order" not in {t.name for t in tools}


def test_mcp_server_status_marks_unreachable():
    from app.mcp.client import mcp_server_status

    client = FakeClient({"logistics": [FakeTool("query_logistics")]}, fail={"aftersale"})
    status = asyncio.run(mcp_server_status(client=client))
    assert status == {"logistics": True, "aftersale": False}
