"""Langfuse 可观测性集成(ch09)。

env 未配 ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_PUBLIC_KEY`` 时静默跳过(返回 None),
任何路径不因 Langfuse 崩。挂一次 ``CallbackHandler`` 到 graph config 即可拿到
完整 trace 树(节点 prompt / 工具调用 / 检索 / token / 耗时)。
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings


@lru_cache(maxsize=1)
def langfuse_enabled() -> bool:
    s = get_settings()
    return bool(s.langfuse_secret_key and s.langfuse_public_key)


def get_langfuse_handler():
    """返回 Langfuse CallbackHandler;未配置返回 None。"""
    if not langfuse_enabled():
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def langfuse_trace_name(conv_id, intent: str | None) -> str:
    """给 trace 起个可读名(会话 + 意图),便于在 Langfuse 里筛。"""
    return f"conv={conv_id}" + (f" intent={intent}" if intent else "")


def langfuse_trace_url(trace_id: str | None) -> str:
    """返回 trace 在 Langfuse UI 的链接;拿不到返回空串。"""
    if not trace_id:
        return ""
    try:
        from langfuse import get_client

        return get_client().get_trace_url(trace_id)
    except Exception:  # noqa: BLE001
        return ""
