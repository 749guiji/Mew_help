"""Tests for the unified tool executor (ch08)."""

import asyncio

import pytest

from app.tools.executor import ToolExecutor
from app.tools.registry import RegisteredTool


def _rt(name, read_only=True, json_schema=None, impl=None, source="builtin"):
    return RegisteredTool(
        name=name,
        description="d",
        json_schema=json_schema or {"type": "object", "properties": {}},
        read_only=read_only,
        source=source,
        mcp_server="srv" if source == "mcp" else None,
        impl=impl or _Ok(),
    )


class _Ok:
    def __init__(self, result="fine"):
        self.result = result
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        return self.result


class _TupleOk:
    async def ainvoke(self, args):
        return ("内容", None)


class _TimeoutThenOk:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        if self.calls == 1:
            raise asyncio.TimeoutError()
        return "ok"


class _Boom:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        raise RuntimeError("boom")


class _Slow:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        await asyncio.sleep(10)
        return "late"


def _recorder():
    calls = []

    def audit(**kw):
        calls.append(kw)

    return calls, audit


def _run(coro):
    return asyncio.run(coro)


def test_validation_error_returns_kind_and_audits():
    calls, audit = _recorder()
    ex = ToolExecutor(retries=0, audit=audit)
    tool = _rt("q", json_schema={"type": "object", "required": ["order_id"]})
    r = _run(ex.execute(tool, {}))
    assert r.kind == "validation_error"
    assert calls[-1]["status"] == "校验拦下"


def test_read_success_unwraps_tuple_and_audits():
    calls, audit = _recorder()
    ex = ToolExecutor(audit=audit)
    tool = _rt("q", impl=_TupleOk())
    r = _run(ex.execute(tool, {}))
    assert r.kind == "ok" and r.content == "内容"
    assert calls[-1]["status"] == "成功"


def test_transient_retries_then_recovers():
    calls, audit = _recorder()
    impl = _TimeoutThenOk()
    ex = ToolExecutor(retries=1, audit=audit)
    tool = _rt("q", impl=impl)
    r = _run(ex.execute(tool, {}))
    assert r.kind == "ok" and r.content == "ok"
    assert impl.calls == 2
    assert calls[-1]["retry_count"] == 1


def test_business_empty_no_retry():
    calls, audit = _recorder()
    impl = _Ok(result="")
    ex = ToolExecutor(retries=2, audit=audit)
    tool = _rt("q", impl=impl)
    r = _run(ex.execute(tool, {}))
    assert r.kind == "ok" and r.content == ""
    assert impl.calls == 1  # 正常返回(空)不重试
    assert calls[-1]["status"] == "成功"


def test_non_transient_no_retry():
    calls, audit = _recorder()
    impl = _Boom()
    ex = ToolExecutor(retries=2, audit=audit)
    tool = _rt("q", impl=impl)
    r = _run(ex.execute(tool, {}))
    assert r.kind == "error"
    assert impl.calls == 1  # 非暂态不重试
    assert calls[-1]["status"] == "失败"


def test_write_needs_confirmation_no_audit():
    calls, audit = _recorder()
    ex = ToolExecutor(audit=audit)
    tool = _rt("create_ticket", read_only=False, impl=_Ok(result="T123"))
    args = {"description": "x", "ticket_type": "售后"}
    r = _run(ex.execute(tool, args))
    assert r.kind == "needs_confirmation"
    assert r.draft == args
    assert calls == []  # 确认前不审计


def test_write_confirmed_timeout_no_retry():
    calls, audit = _recorder()
    impl = _Slow()
    ex = ToolExecutor(timeout=0.01, audit=audit)
    tool = _rt("create_ticket", read_only=False, impl=impl)
    with pytest.raises(asyncio.TimeoutError):
        _run(ex.execute_confirmed(tool, {"description": "x", "ticket_type": "售后"}))
    assert impl.calls == 1  # 写不重试
    assert calls[-1]["status"] == "超时"
    assert calls[-1]["retry_count"] == 0


def test_mcp_write_not_allowlisted_denied():
    calls, audit = _recorder()
    ex = ToolExecutor(write_allowlist=set(), audit=audit)
    tool = _rt("mcp_write", read_only=False, source="mcp", impl=_Ok())
    r = _run(ex.execute(tool, {}))
    assert r.kind == "permission_denied"
    assert calls[-1]["status"] == "权限拒绝"
