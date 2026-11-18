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

                results: list[tuple[dict, str]] = []
                refusal_text: str | None = None
                token = current_conversation_id.set(conv_id)
                try:
                    for tc in ai_msg.tool_calls:
                        yield {"type": "tool", "name": tc["name"], "status": "running"}
                        result = await self._executor.execute(tc["name"], tc["args"])
                        status = "error" if result.startswith("[工具错误]") else "done"
                        yield {"type": "tool", "name": tc["name"], "status": status}

                        parsed = None
                        if tc["name"] == "query_faq":
                            try:
                                parsed = json.loads(result)
                            except (json.JSONDecodeError, TypeError):
                                parsed = None

                        if parsed is not None and parsed.get("refused") is True:
                            # Retrieval refused: no ToolMessage, canned refusal,
                            # log into the low-confidence pool, skip convergence.
                            if refusal_text is None:
                                yield {"type": "token", "content": REFUSAL_TOKEN}
                                refusal_text = REFUSAL_TOKEN
                            with SessionLocal() as s:
                                log_low_confidence(
                                    s,
                                    conversation_id=conv_id,
                                    raw_question=message,
                                    source="retrieval_low_conf",
                                    reason=None,
                                )
                            continue

                        if parsed is not None and parsed.get("refused") is False:
                            citations = parsed.get("citations") or []
                            yield {"type": "citations", "citations": citations}
                            evidence = parsed.get("evidence") or ""
                            check = self_check(self._llm, message, evidence)
                            if not check.sufficient:
                                if refusal_text is None:
                                    yield {"type": "token", "content": REFUSAL_TOKEN}
                                    refusal_text = REFUSAL_TOKEN
                                with SessionLocal() as s:
                                    log_low_confidence(
                                        s,
                                        conversation_id=conv_id,
                                        raw_question=message,
                                        source="self_check",
                                        reason=check.reason,
                                    )
                                continue
                            # Sufficient: feed the evidence text (not the JSON).
                            result = evidence

                        results.append((tc, result))
                        messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tc["id"],
                                name=tc["name"],
                            )
                        )
                finally:
                    current_conversation_id.reset(token)

                if refusal_text is not None:
                    self._save_message(conv_id, role="user", content=message)
                    self._save_message(conv_id, role="assistant", content=refusal_text)
                    reply = refusal_text
                else:
                    self._save_message(conv_id, role="user", content=message)
                    self._save_message(
                        conv_id,
                        role="assistant",
                        content=None,
                        tool_calls=ai_msg.tool_calls,
                    )
                    for tc, result in results:
                        self._save_message(
                            conv_id, role="tool", content=result, tool_call_id=tc["id"]
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
