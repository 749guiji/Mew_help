from collections.abc import Sequence
from functools import lru_cache

import tiktoken
from langchain_core.messages import BaseMessage


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _message_text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    text = "".join(_message_text(m) for m in messages)
    try:
        return len(_get_encoding().encode(text))
    except Exception:
        return len(text) // 4


def trim_history(
    messages: Sequence[BaseMessage], max_tokens: int, keep_last: int
) -> list[BaseMessage]:
    result = list(messages)
    if keep_last < len(result):
        result = result[-keep_last:]
    while count_tokens(result) > max_tokens:
        result.pop(0)
    return result
