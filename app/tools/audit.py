"""Tool-call audit logger (ch08).

审计写入绝不能反过来拦断工具执行:所有异常都被吞掉、只记日志。内置和 MCP
工具都记,被权限拒、被校验拦的调用同样落一条。
"""

from __future__ import annotations

import logging

from app.db import SessionLocal
from app.db.models import ToolAuditLog

logger = logging.getLogger(__name__)

_RESULT_SUMMARY_MAX = 2000


def log_tool_audit(
    *,
    conversation_id: int | None = None,
    tool_call_id: str | None = None,
    tool_name: str,
    tool_source: str,
    mcp_server: str | None = None,
    arguments: dict | None = None,
    result_summary: str | None = None,
    status: str,
    error_message: str | None = None,
    retry_count: int = 0,
    duration_ms: int | None = None,
) -> None:
    try:
        if result_summary:
            result_summary = result_summary[:_RESULT_SUMMARY_MAX]
        with SessionLocal() as s:
            s.add(
                ToolAuditLog(
                    conversation_id=conversation_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_source=tool_source,
                    mcp_server=mcp_server,
                    arguments=arguments,
                    result_summary=result_summary,
                    status=status,
                    error_message=error_message,
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
            )
            s.commit()
    except Exception as e:  # noqa: BLE001 - 审计失败绝不能拦工具执行
        logger.warning("tool audit write failed (tool=%s status=%s): %s", tool_name, status, e)
