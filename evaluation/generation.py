"""Answer-coverage LLM judge for the generation segment of the eval.

``judge_answer_coverage`` asks the LLM which of the standard answer points a
generated answer actually covers. Unlike ``self_check``/``judge_faithfulness``
(which are gates and fall back permissively), this is a *measurement*: when the
structured call fails to emit a ``tool_call`` it returns ``None`` so the eval
harness skips the item instead of fabricating a coverage score.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.prompts.system import ANSWER_COVERAGE_SYSTEM_PROMPT


class AnswerCoverageResult(BaseModel):
    """Zero-based indices of the standard answer points the answer covers."""

    covered: list[int]


def judge_answer_coverage(
    llm, question: str, answer: str, expect_points: list[str]
) -> AnswerCoverageResult | None:
    """Ask ``llm`` which ``expect_points`` (0-based) ``answer`` covers.

    Returns ``None`` when the structured output is not an
    :class:`AnswerCoverageResult` (e.g. Ollama FC miss), so callers can skip the
    item rather than count a failed judge as zero coverage.
    """
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(expect_points))
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_COVERAGE_SYSTEM_PROMPT),
            (
                "human",
                "用户问题：{question}\n\n助手回答：{answer}\n\n标准答案要点：\n{points}",
            ),
        ]
    ) | llm.with_structured_output(AnswerCoverageResult, method="function_calling")
    result = chain.invoke({"question": question, "answer": answer, "points": numbered})
    if isinstance(result, AnswerCoverageResult):
        return result
    return None
