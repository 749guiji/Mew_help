"""Tests for the ch08 tool registry (RegisteredTool + ToolRegistry)."""

from app.tools.registry import RegisteredTool, ToolRegistry, registry


class _Echo:
    """Minimal impl stand-in for registration tests."""

    async def ainvoke(self, args):
        return args


def test_agent_builtin_excludes_query_faq_and_logistics():
    names = {t.name for t in registry.agent_builtin()}
    assert "query_order" in names and "create_ticket" in names
    assert "query_faq" not in names and "query_logistics" not in names


def test_create_ticket_is_write():
    t = registry.get("create_ticket")
    assert t.read_only is False and t.source == "builtin"


def test_new_tool_registered_is_picked_up():
    r = ToolRegistry()
    r.register(
        RegisteredTool("echo", "desc", {"type": "object"}, True, "builtin", None, _Echo())
    )
    assert "echo" in {t.name for t in r.agent_builtin()}
