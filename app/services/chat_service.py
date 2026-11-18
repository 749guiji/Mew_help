import json
import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage

from app.core.config import get_settings
from app.core.llm import get_llm
from app.db import SessionLocal
from app.db.models import Conversation, Message
from app.prompts.system import build_chat_messages
from app.services.generation_quality import log_low_confidence, self_check
from app.services.history import trim_history
from app.tools import ALL_TOOLS
from app.tools.context import current_conversation_id
from app.tools.executor import ToolExecutor

REFUSAL_TOKEN = "抱歉，我目前的知识库中没有足够信息回答您的问题，已为您记录并转人工处理。"

logger = logging.getLogger(__name__)


def _log_low_confidence(
    conv_id: int | None, raw_question: str, source: str, reason: str | None
) -> None:
    """Log a low-confidence question, swallowing DB errors like ``_save_message``."""
    try:
        with SessionLocal() as s:
            log_low_confidence(
                s,
                conversation_id=conv_id,
                raw_question=raw_question,
                source=source,
                reason=reason,
            )
    except Exception as e:
        logger.warning(
            "Failed to log low-confidence question (%s) for conversation %s: %s",
            source,
            conv_id,
            e,
        )


class ChatService:
    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools=None,
        executor: ToolExecutor | None = None,
    ):
        settings = get_settings()
        self._llm = llm if llm is not None else get_llm(settings)
        self._tools = tools if tools is not None else ALL_TOOLS
        self._executor = executor if executor is not None else ToolExecutor()
        self._max_tokens = settings.max_history_tokens
        self._max_messages = settings.max_history_messages
        self._sessions: dict[str, list[BaseMessage]] = {}

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
            logging.getLogger(__name__).warning(
                "Failed to resolve/create conversation for %s: %s", sid, e
            )
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
            logging.getLogger(__name__).warning(
                "Failed to save message (role=%s) for conversation %s: %s",
                role,
                conv_id,
                e,
            )

    async def stream_chat(
        self, session_id: str | None, message: str
    ) -> AsyncIterator[dict[str, str]]:
        try:
            sid = session_id if session_id in self._sessions else uuid4().hex
            history = self._sessions.setdefault(sid, [])

            yield {"type": "meta", "session_id": sid}

            conv_id = self._resolve_conversation(sid)

            messages = build_chat_messages(
                trim_history(history, self._max_tokens, self._max_messages), message
            )

            # Stream the tool-aware model once and aggregate the chunks. Non-tool
            # replies are emitted token-by-token as they arrive; a tool decision is
            # detected only after the stream ends.
            ai_msg: AIMessageChunk | None = None
            async for chunk in self._llm.bind_tools(self._tools).astream(messages):
                ai_msg = chunk if ai_msg is None else ai_msg + chunk
                content = chunk.content
                if content:
                    yield {"type": "token", "content": content}

            if ai_msg is not None and ai_msg.tool_calls:
                messages.append(ai_msg)

                executed: list[tuple[dict, str]] = []
                token = current_conversation_id.set(conv_id)
                try:
                    for tc in ai_msg.tool_calls:
                        yield {"type": "tool", "name": tc["name"], "status": "running"}
                        raw = await self._executor.execute(tc["name"], tc["args"])
                        status = "error" if raw.startswith("[工具错误]") else "done"
                        yield {"type": "tool", "name": tc["name"], "status": status}
                        executed.append((tc, raw))
                finally:
                    current_conversation_id.reset(token)

                # Process each executed tool into its final ToolMessage content and
                # any refusal metadata, then decide whether to short-circuit. A
                # query_faq refusal only short-circuits when every executed tool
                # refused (no other tool produced a result to persist/show).
                contents: list[str] = []
                citations_by_tool: list[list | None] = []
                refusal_sources: list[str | None] = []
                refusal_reasons: list[str | None] = []

                for tc, raw in executed:
                    parsed = None
                    if tc["name"] == "query_faq":
                        try:
                            parsed = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            parsed = None

                    source = None
                    reason = None
                    citations = None
                    content = raw

                    if parsed is not None and parsed.get("refused") is True:
                        source = "retrieval_low_conf"
                    elif parsed is not None and parsed.get("refused") is False:
                        citations = parsed.get("citations") or []
                        evidence = parsed.get("evidence") or ""
                        check = self_check(self._llm, message, evidence)
                        if not check.sufficient:
                            source = "self_check"
                            reason = check.reason
                        else:
                            content = evidence

                    contents.append(content)
                    citations_by_tool.append(citations)
                    refusal_sources.append(source)
                    refusal_reasons.append(reason)

                has_refusal = any(s is not None for s in refusal_sources)
                has_non_refused = any(s is None for s in refusal_sources)
                short_circuit = has_refusal and not has_non_refused

                for citations in citations_by_tool:
                    if citations is not None:
                        yield {"type": "citations", "citations": citations}

                for source, reason in zip(refusal_sources, refusal_reasons):
                    if source is not None:
                        _log_low_confidence(conv_id, message, source, reason)

                if short_circuit:
                    yield {"type": "token", "content": REFUSAL_TOKEN}
                    self._save_message(conv_id, role="user", content=message)
                    self._save_message(conv_id, role="assistant", content=REFUSAL_TOKEN)
                    reply = REFUSAL_TOKEN
                else:
                    results: list[tuple[dict, str]] = []
                    for (tc, raw), content in zip(executed, contents):
                        results.append((tc, content))
                        messages.append(
                            ToolMessage(
                                content=content,
                                tool_call_id=tc["id"],
                                name=tc["name"],
                            )
                        )

                    self._save_message(conv_id, role="user", content=message)
                    self._save_message(
                        conv_id,
                        role="assistant",
                        content=None,
                        tool_calls=ai_msg.tool_calls,
                    )
                    for tc, content in results:
                        self._save_message(
                            conv_id, role="tool", content=content, tool_call_id=tc["id"]
                        )

                    # Final convergence stream uses the UNBOUND model so the model
                    # cannot trigger a second tool call in the same turn.
                    reply_parts: list[str] = []
                    async for chunk in self._llm.astream(messages):
                        content = chunk.content
                        if not content:
                            continue
                        reply_parts.append(content)
                        yield {"type": "token", "content": content}
                    reply = "".join(reply_parts)

                    self._save_message(conv_id, role="assistant", content=reply)
            else:
                reply = ai_msg.content if ai_msg is not None else ""
                self._save_message(conv_id, role="user", content=message)
                self._save_message(conv_id, role="assistant", content=reply)

            history.append(HumanMessage(content=message))
            history.append(AIMessage(content=reply or ""))

            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
