"""Query understanding: LLM rewrite + retrieval-time synonym expansion.

Before a user question hits hybrid retrieval, it goes through two transforms:

- ``rewrite_query`` turns a colloquial question into a standard, searchable form
  (plus an inferred category) via a structured ``with_structured_output(...,
  method="function_calling")`` chain. ``method="function_calling"`` is explicit
  because langchain 1.x otherwise defaults to ``json_schema``, which silently
  fails on Ollama.
- ``expand_synonyms`` is a pure in-memory lookup over :data:`SYNONYM_DICT` that
  appends matching synonyms to the query text so BM25 can match more tokens. It
  only returns a text string: it never writes to any store and never creates KB
  chunks (retrieval-side only, no duplicate storage).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.prompts.system import QUERY_REWRITE_SYSTEM_PROMPT


class QueryUnderstanding(BaseModel):
    standard_question: str
    category: str | None = None


SYNONYM_DICT: dict[str, list[str]] = {
    "退款": ["退钱", "退货退款"],
    "运费": ["邮费"],
    "发货": ["寄出", "物流"],
}


def rewrite_query(llm, question: str) -> QueryUnderstanding:
    """Rewrite ``question`` into a standard form + inferred category.

    ``llm`` must expose ``with_structured_output(model, method=...)`` (as the
    langchain chat models do). The result is returned as a :class:`QueryUnderstanding`.
    """
    chain = ChatPromptTemplate.from_messages(
        [("system", QUERY_REWRITE_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm.with_structured_output(QueryUnderstanding, method="function_calling")
    result = chain.invoke({"input": question})
    if isinstance(result, QueryUnderstanding):
        return result
    # Ollama FC can fail to emit a tool_call (e.g. qwen3:8b), yielding None.
    # Never return None: fall back to the original question so retrieval still
    # proceeds with a usable query instead of raising on `.standard_question`.
    return QueryUnderstanding(standard_question=question, category=None)


def expand_synonyms(text: str) -> str:
    """Append synonyms for every ``SYNONYM_DICT`` key found in ``text``.

    Returns ``text`` unchanged when nothing matches; otherwise the original text
    followed by the matched synonyms (space-separated), so BM25 can match more
    tokens without the KB ever storing the expanded form.
    """
    extra: list[str] = []
    for key, synonyms in SYNONYM_DICT.items():
        if key in text:
            extra.extend(synonyms)
    if not extra:
        return text
    return f"{text} {' '.join(extra)}"
