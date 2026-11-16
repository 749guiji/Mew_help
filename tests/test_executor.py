"""Tests for ToolExecutor: timeout, retry, and error handling."""

import asyncio

from app.tools import TOOL_REGISTRY
from app.tools.executor import ToolExecutor


class _Ok:
    name = "ok"

    async def ainvoke(self, args):
        return "fine"


class _Boom:
    name = "boom"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        raise RuntimeError("x")


class _Flaky:
    name = "flaky"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return "recovered"


class _Slow:
    name = "slow"

    async def ainvoke(self, args):
        await asyncio.sleep(10)
        return "late"


def test_execute_success():
    ex = ToolExecutor(registry={"ok": _Ok()})
    assert asyncio.run(ex.execute("ok", {})) == "fine"


def test_execute_retries_then_error_string():
    boom = _Boom()
    ex = ToolExecutor(registry={"boom": boom}, retries=1)
    out = asyncio.run(ex.execute("boom", {}))
    assert out.startswith("[工具错误]")
    assert boom.calls == 2


def test_execute_retries_then_succeeds():
    flaky = _Flaky()
    ex = ToolExecutor(registry={"flaky": flaky}, retries=1)
    out = asyncio.run(ex.execute("flaky", {}))
    assert out == "recovered"
    assert flaky.calls == 2


def test_execute_timeout_returns_error_string():
    ex = ToolExecutor(registry={"slow": _Slow()}, timeout=0.01, retries=0)
    out = asyncio.run(ex.execute("slow", {}))
    assert out.startswith("[工具错误]")


def test_default_registry_is_tool_registry():
    ex = ToolExecutor()
    assert ex.registry is TOOL_REGISTRY
    assert ex.timeout == 5.0
    assert ex.retries == 2


def test_execute_unknown_tool_returns_error_string():
    ex = ToolExecutor(registry={})
    out = asyncio.run(ex.execute("nope", {}))
    assert out.startswith("[工具错误]")
