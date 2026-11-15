from collections.abc import AsyncIterator
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import get_settings
from app.core.llm import get_llm
from app.prompts.system import build_chat_messages
from app.services.history import trim_history


class ChatService:
    def __init__(self, llm: BaseChatModel | None = None):
        settings = get_settings()
        self._llm = llm if llm is not None else get_llm(settings)
        self._max_tokens = settings.max_history_tokens
        self._max_messages = settings.max_history_messages
        self._sessions: dict[str, list[BaseMessage]] = {}

    async def stream_chat(
        self, session_id: str | None, message: str
    ) -> AsyncIterator[dict[str, str]]:
        try:
            sid = session_id if session_id in self._sessions else uuid4().hex
            history = self._sessions.setdefault(sid, [])

            yield {"type": "meta", "session_id": sid}

            messages = build_chat_messages(
                trim_history(history, self._max_tokens, self._max_messages), message
            )

            reply_parts: list[str] = []
            async for chunk in self._llm.astream(messages):
                content = chunk.content
                if not content:
                    continue
                reply_parts.append(content)
                yield {"type": "token", "content": content}

            history.append(HumanMessage(content=message))
            history.append(AIMessage(content="".join(reply_parts)))

            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
