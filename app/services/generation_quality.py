"""Generation quality controls for the RAG answer path.

The chat tool path returns retrieved evidence, but before that evidence becomes
a user-facing answer we (a) format it with numbered citations, (b) let an LLM
self-check whether the evidence actually supports an answer, and (c) log
questions that cannot be answered into ``low_confidence_questions`` for the
manual review pool.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.db.models import LowConfidenceQuestion
from app.kb.hybrid import ScoredChunk
from app.prompts.system import SELF_CHECK_SYSTEM_PROMPT


class SelfCheckResult(BaseModel):
    """Whether the retrieved evidence is sufficient to answer the question."""

    sufficient: bool
    reason: str


def self_check(llm, question: str, evidence: str) -> SelfCheckResult:
    """Ask ``llm`` whether ``evidence`` supports an answer to ``question``.

    Uses ``with_structured_output(..., method="function_calling")`` explicitly
    (langchain 1.x otherwise defaults to ``json_schema``, which silently fails
    on Ollama). The chain output is coerced back to :class:`SelfCheckResult` so
    callers can rely on attribute access.
    """
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", SELF_CHECK_SYSTEM_PROMPT),
            ("human", "用户问题：{question}\n\n检索到的证据：\n{evidence}"),
        ]
    ) | llm.with_structured_output(SelfCheckResult, method="function_calling")
    result = chain.invoke({"question": question, "evidence": evidence})
    if isinstance(result, SelfCheckResult):
        return result
    # Ollama FC can fail to emit a tool_call (e.g. qwen3:8b), yielding None.
    # Never crash the answer path: the evidence already passed the retrieval
    # threshold, so don't refuse just because the checker couldn't run.
    return SelfCheckResult(sufficient=True, reason="")


def log_low_confidence(
    session,
    *,
    conversation_id: int | None,
    raw_question: str,
    source: str,
    reason: str | None = None,
) -> None:
    """Insert one ``LowConfidenceQuestion`` row and commit ``session``."""
    session.add(
        LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
        )
    )
    session.commit()


def format_evidence(chunks: list[ScoredChunk]) -> str:
    """Render numbered evidence lines: ``[1] {section_path or category}: {answer}``."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        section = chunk.section_path or chunk.category
        lines.append(f"[{i}] {section}: {chunk.answer}")
    return "\n".join(lines)


def build_citations(chunks: list[ScoredChunk]) -> list[dict]:
    """Build the citation payload surfaced to the client via the SSE event."""
    return [
        {
            "n": i,
            "chunk_id": chunk.chunk_id,
            "section_path": chunk.section_path,
            "answer_snippet": chunk.answer[:200],
        }
        for i, chunk in enumerate(chunks, start=1)
    ]
