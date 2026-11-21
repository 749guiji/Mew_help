"""Unified tool executor (ch08): validation → permission → execute → triage → audit.

所有工具调用(内置 + MCP)都走这里。读工具超时/网络暂态重试、业务空不重试;
写工具(create_ticket)不直接执行,返回 ``needs_confirmation`` 等前端确认,确认后
走 ``execute_confirmed``(不自动重试,``retry_count`` 恒 0)。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

import jsonschema

from app.core.config import get_settings
from app.tools.audit import log_tool_audit
from app.tools.context import current_conversation_id
from app.tools.registry import RegisteredTool

# 网络暂态异常:重试只给这类。Task 5 接入 MCP 后把 MCP 传输/会话异常加进来。
TRANSIENT_EXCEPTIONS = (asyncio.TimeoutError,)


@dataclass
class ToolResult:
    kind: str  # ok | validation_error | permission_denied | needs_confirmation | error
    content: str
    draft: dict | None = None


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, TRANSIENT_EXCEPTIONS)


def _stringify(result) -> str:
    """把工具返回值归一成给模型看的字符串(解包 (content, artifact) 元组/块列表)。"""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class ToolExecutor:
    def __init__(
        self,
        *,
        timeout: float | None = None,
        retries: int | None = None,
        write_allowlist: set[str] | None = None,
        audit: Callable | None = None,
    ) -> None:
        settings = get_settings()
        self.timeout = timeout if timeout is not None else settings.tool_timeout_seconds
        self.retries = retries if retries is not None else settings.tool_retries
        self.write_allowlist = (
            write_allowlist if write_allowlist is not None else set(settings.mcp_write_tools)
        )
        self.audit = audit if audit is not None else log_tool_audit

    def _record(
        self,
        tool: RegisteredTool,
        args: dict,
        tool_call_id: str | None,
        conversation_id: int | None,
        *,
        status: str,
        result_summary: str | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
        duration_ms: int | None = None,
    ) -> None:
        self.audit(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            tool_name=tool.name,
            tool_source=tool.source,
            mcp_server=tool.mcp_server,
            arguments=args,
            result_summary=result_summary,
            status=status,
            error_message=error_message,
            retry_count=retry_count,
            duration_ms=duration_ms,
        )

    async def execute(
        self,
        tool: RegisteredTool,
        args: dict,
        *,
        tool_call_id: str | None = None,
        conversation_id: int | None = None,
    ) -> ToolResult:
        if not isinstance(args, dict):
            args = {}

        # 1. 参数校验(JSON Schema)
        try:
            jsonschema.validate(args, tool.json_schema)
        except jsonschema.ValidationError as e:
            self._record(
                tool, args, tool_call_id, conversation_id,
                status="校验拦下", error_message=e.message, retry_count=0,
            )
            return ToolResult("validation_error", f"参数校验失败:{e.message}")

        # 2. 权限:写操作由引擎把门
        if not tool.read_only:
            if tool.source == "mcp" and tool.name not in self.write_allowlist:
                msg = f"工具 {tool.name} 是写操作,未获授权,已拒绝调用。"
                self._record(
                    tool, args, tool_call_id, conversation_id,
                    status="权限拒绝", error_message=msg, retry_count=0,
                )
                return ToolResult("permission_denied", msg)
            # 内置写(create_ticket):不直接执行,等前端确认
            return ToolResult("needs_confirmation", "", draft=args)

        # 3. 读:执行 + 超时/重试/三分类
        return await self._run_read(tool, args, tool_call_id, conversation_id)

    async def _run_read(
        self,
        tool: RegisteredTool,
        args: dict,
        tool_call_id: str | None,
        conversation_id: int | None,
    ) -> ToolResult:
        start = time.monotonic()
        last_exc: Exception | None = None
        attempts = 0
        for _ in range(self.retries + 1):
            attempts += 1
            try:
                token = current_conversation_id.set(conversation_id)
                try:
                    raw = await asyncio.wait_for(tool.impl.ainvoke(args), self.timeout)
                finally:
                    current_conversation_id.reset(token)
                content = _stringify(raw)
                self._record(
                    tool, args, tool_call_id, conversation_id,
                    status="成功", result_summary=content,
                    retry_count=attempts - 1, duration_ms=_elapsed_ms(start),
                )
                return ToolResult("ok", content)
            except Exception as e:  # noqa: BLE001 - 统一分诊
                last_exc = e
                if not _is_transient(e):
                    break  # 非暂态(业务/DB 等)不重试

        status = "超时" if isinstance(last_exc, asyncio.TimeoutError) else "失败"
        self._record(
            tool, args, tool_call_id, conversation_id,
            status=status, error_message=str(last_exc),
            retry_count=attempts - 1, duration_ms=_elapsed_ms(start),
        )
        return ToolResult("error", f"[工具错误] {last_exc}")

    async def execute_confirmed(
        self,
        tool: RegisteredTool,
        args: dict,
        *,
        tool_call_id: str | None = None,
        conversation_id: int | None = None,
    ) -> str:
        """写工具确认后执行:超时兜底、不自动重试(retry_count 恒 0),失败抛异常。"""
        start = time.monotonic()
        try:
            token = current_conversation_id.set(conversation_id)
            try:
                raw = await asyncio.wait_for(tool.impl.ainvoke(args), self.timeout)
            finally:
                current_conversation_id.reset(token)
            content = _stringify(raw)
            self._record(
                tool, args, tool_call_id, conversation_id,
                status="成功", result_summary=content,
                retry_count=0, duration_ms=_elapsed_ms(start),
            )
            return content
        except Exception as e:  # noqa: BLE001
            status = "超时" if isinstance(e, asyncio.TimeoutError) else "失败"
            self._record(
                tool, args, tool_call_id, conversation_id,
                status=status, error_message=str(e),
                retry_count=0, duration_ms=_elapsed_ms(start),
            )
            raise
