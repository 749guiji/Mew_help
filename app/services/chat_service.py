"""Chat service backed by the ch06 LangGraph orchestration graph.

The graph routes deterministically (intent → 5 outlets) and runs the ReAct agent /
fixed replies; this service streams SSE events from the graph's final state and
persists messages to MySQL. The refund sub-flow pauses via LangGraph ``interrupt``
and resumes via ``Command(resume)`` (same ``thread_id`` = session_id).
"""

import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.db import SessionLocal
from app.db.models import Conversation, Message
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

_GRAPH = None


def _get_graph():
    """Cached compiled graph (its InMemorySaver persists across turns)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


class ChatService:
    def __init__(self, graph=None):
        self._graph = graph  # injected for tests
        self._sessions: set[str] = set()

    def _resolve_conversation(self, user_id: str, conversation_id: int | None = None) -> int | None:
        """按 user_id 建/查会话;给定 conversation_id 则续用(thread_id=conversation_id)。"""
        try:
            with SessionLocal() as s:
                if conversation_id is not None:
                    conv = s.get(Conversation, conversation_id)
                    if conv is not None and conv.user_id == user_id:
                        return conv.id
                conv = Conversation(user_id=user_id, status="进行中")
                s.add(conv)
                s.commit()
                s.refresh(conv)
                return conv.id
        except Exception as e:
            logger.warning("Failed to resolve/create conversation for %s: %s", user_id, e)
            return None

    def _load_history(self, conv_id: int) -> list:
        """回灌会话历史(user/assistant 原文,工具结果不落表故不含)。"""
        try:
            with SessionLocal() as s:
                rows = (
                    s.query(Message)
                    .filter(Message.conversation_id == conv_id)
                    .order_by(Message.id)
                    .all()
                )
                return [(m.role, m.content) for m in rows if m.content]
        except Exception as e:
            logger.warning("Failed to load history conv=%s: %s", conv_id, e)
            return []

    def _save_message(
        self,
        conv_id: int | None,
        *,
        role: str,
        content: str | None = None,
        tool_calls=None,
        tool_call_id: str | None = None,
    ) -> None:
        if conv_id is None:
            return
        try:
            with SessionLocal() as s:
                s.add(
                    Message(
                        conversation_id=conv_id,
                        role=role,
                        content=content,
                        tool_calls=tool_calls,
                        tool_call_id=tool_call_id,
                    )
                )
                s.commit()
        except Exception as e:
            logger.warning("Failed to save message (role=%s): %s", role, e)

    async def _stream_invoke(
        self, conv_id: int, user_id: str, invoke_input
    ) -> AsyncIterator[dict]:
        """ainvoke(正常或 Command(resume))后,吐本轮事件;遇中断发 interrupt 帧。

        thread_id=conversation_id;checkpoint 空则从 MySQL 回灌 user/assistant 历史。
        """
        graph = self._graph if self._graph is not None else _get_graph()
        config = {"configurable": {"thread_id": str(conv_id)}}
        if isinstance(invoke_input, dict):
            invoke_input = {
                **invoke_input,
                "user_id": user_id,
                "conversation_id": conv_id,
            }
            before = await graph.aget_state(config)
            if not before.values.get("messages"):
                # 空 checkpoint(切会话/重启)→ 回灌历史
                invoke_input["messages"] = (
                    self._load_history(conv_id) + invoke_input.get("messages", [])
                )
            before_count = len(before.values.get("messages", []))
        else:
            before = await graph.aget_state(config)
            before_count = len(before.values.get("messages", []))
        result = await graph.ainvoke(invoke_input, config=config)

        # 中断帧:退款子流程缺订单号(select_order)或建工单预览(ticket_preview)
        intr = result.get("__interrupt__")
        if intr:
            payload = intr[0].value
            frame = {"type": "interrupt", "kind": payload.get("type", "")}
            if frame["kind"] == "select_order":
                frame["orders"] = payload.get("orders", [])
            elif frame["kind"] == "ticket_preview":
                frame["draft"] = payload.get("draft", {})
            yield frame
            return

        # 只取本轮新增的消息,避免回放历史轮次的工具徽章 / 旧答案
        new_msgs = result.get("messages", [])[before_count:]

        # 工具帧(本轮 Agent 已执行完的工具,回放给前端徽章)
        for m in new_msgs:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    yield {"type": "tool", "name": tc["name"], "status": "done"}

        # 引用帧(知识类检索块)
        chunks = result.get("retrieval_chunks") or []
        if chunks:
            from app.services.generation_quality import build_citations

            yield {"type": "citations", "citations": build_citations(chunks)}

        # 答案帧(本轮最后一个有内容的 AIMessage)
        answer = ""
        for m in reversed(new_msgs):
            if isinstance(m, AIMessage) and m.content:
                answer = m.content
                break
        if answer:
            yield {"type": "token", "content": answer}

        # 转人工/建工单建议帧
        handoff = result.get("handoff_suggestions") or []
        if handoff:
            yield {"type": "handoff", "suggestions": handoff}

        # 动作帧(退款表单 refund_form,来自 submit_refund 拦截)
        actions = result.get("suggested_actions") or []
        if actions:
            yield {"type": "actions", "items": actions}

        yield {"type": "done"}

    async def stream_chat(
        self, user_id: str | None, message: str, conversation_id: int | None = None
    ) -> AsyncIterator[dict]:
        try:
            user_id = user_id or uuid4().hex
            conv_id = self._resolve_conversation(user_id, conversation_id)
            yield {"type": "meta", "session_id": str(conv_id)}

            answer = ""
            async for ev in self._stream_invoke(
                conv_id, user_id, {"messages": [("user", message)], "raw_query": message}
            ):
                if ev["type"] == "token":
                    answer = ev["content"]
                yield ev

            self._save_message(conv_id, role="user", content=message)
            self._save_message(conv_id, role="assistant", content=answer or None)
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    async def stream_resume(
        self, conversation_id: int, payload
    ) -> AsyncIterator[dict]:
        """续跑暂停流:payload 透传给 Command(resume)。

        订单选择=order_id 字符串,建工单确认={"confirm": bool}。
        """
        try:
            with SessionLocal() as s:
                conv = s.get(Conversation, conversation_id)
                user_id = conv.user_id if conv else "guest"
            async for ev in self._stream_invoke(conversation_id, user_id, Command(resume=payload)):
                yield ev
        except Exception as e:
            yield {"type": "error", "message": str(e)}
