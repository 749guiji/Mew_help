"""Runtime context shared by tool executors."""

from contextvars import ContextVar

current_conversation_id: ContextVar[int | None] = ContextVar(
    "current_conversation_id", default=None
)
