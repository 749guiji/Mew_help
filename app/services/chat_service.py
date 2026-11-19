"""Chat service backed by the ch05 LangGraph orchestration graph.

The graph routes deterministically (intent → route) and runs the ReAct agent /
fixed replies; this service is a thin adapter that streams SSE events from the
graph's final state and persists messages to MySQL.
"""

import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from langchain_core.messages import AIMessage

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

    def _resolve_conversation(self, sid: str) -> int | None:
        try:
            with SessionLocal() as s:
                conv = (
                    s.query(Conversation)
                    .filter(Conversation.user_id == sid)
                    .first()
                )
                if conv is None:
                    conv = Conversation(user_id=sid, status="进行中")
                    s.add(conv)
                    s.commit()
                    s.refresh(conv)
                return conv.id
        except Exception as e:
            logger.warning("Failed to resolve/create conversation for %s: %s", sid, e)
            return None

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

    async def stream_chat(
        self, session_id: str | None, message: str
    ) -> AsyncIterator[dict[str, str]]:
        try:
            sid = session_id if session_id in self._sessions else uuid4().hex
            self._sessions.add(sid)

            yield {"type": "meta", "session_id": sid}

            conv_id = self._resolve_conversation(sid)

            graph = self._graph if self._graph is not None else _get_graph()
            result = await graph.ainvoke(
                {"messages": [("user", message)], "raw_query": message},
                config={"configurable": {"thread_id": sid}},
            )

            msgs = result.get("messages", [])

            # 工具帧(Agent 已执行完的工具,回放给前端徽章)
            for m in msgs:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        yield {"type": "tool", "name": tc["name"], "status": "done"}

            # 引用帧(知识类检索块)
            chunks = result.get("retrieval_chunks") or []
            if chunks:
                from app.services.generation_quality import build_citations

                yield {"type": "citations", "citations": build_citations(chunks)}

            # 答案帧(最后一个有内容的 AIMessage)
            answer = ""
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content:
                    answer = m.content
                    break
            if answer:
                yield {"type": "token", "content": answer}

            # 转人工/建工单建议帧
            handoff = result.get("handoff_suggestions") or []
            if handoff:
                yield {"type": "handoff", "suggestions": handoff}

            self._save_message(conv_id, role="user", content=message)
            self._save_message(conv_id, role="assistant", content=answer or None)

            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
