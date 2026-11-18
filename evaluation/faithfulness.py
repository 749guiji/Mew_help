"""Faithfulness judge + ``faith_cases`` upsert ledger for the generation segment.

The generation segment of the ch04 eval generates an answer from the hybrid+rerank
evidence, then asks ``judge_faithfulness`` whether every factual claim in that
answer is supported by the cited evidence. Unfaithful answers are recorded in the
``faith_cases`` table via :func:`record_faith_case`, which upserts by ``eval_id``
so a repeatedly-bad case accumulates ``seen_count`` instead of duplicating rows.
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.db.models import FaithCase
from app.prompts.system import FAITHFULNESS_SYSTEM_PROMPT


class FaithfulnessResult(BaseModel):
    """Whether every factual claim in the answer is supported by the citations."""

    faithful: bool
    reason: str


def judge_faithfulness(
    llm, question: str, answer: str, citations: list[dict]
) -> FaithfulnessResult:
    """Ask ``llm`` whether ``answer`` is faithful to ``citations``.

    Uses ``with_structured_output(..., method="function_calling")`` explicitly
    (langchain 1.x otherwise defaults to ``json_schema``, which silently fails
    on Ollama). The chain output is coerced back to :class:`FaithfulnessResult`
    so callers can rely on attribute access.
    """
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", FAITHFULNESS_SYSTEM_PROMPT),
            (
                "human",
                "用户问题：{question}\n\n助手回答：{answer}\n\n引用证据（citations）：\n{citations}",
            ),
        ]
    ) | llm.with_structured_output(FaithfulnessResult, method="function_calling")
    result = chain.invoke(
        {
            "question": question,
            "answer": answer,
            "citations": json.dumps(citations, ensure_ascii=False),
        }
    )
    if isinstance(result, FaithfulnessResult):
        return result
    return FaithfulnessResult.model_validate(result)


def record_faith_case(
    session,
    *,
    eval_id,
    bucket,
    query,
    strategy="hybrid_rerank",
    answer,
    reason,
    citations,
    judge_model,
) -> None:
    """Upsert one ``FaithCase`` row by ``eval_id`` and commit ``session``.

    A new ``eval_id`` inserts a fresh row (``seen_count=1``, ``status=未解决``).
    An existing ``eval_id`` refreshes the answer/reason/citations/judge_model
    snapshot, bumps ``seen_count``, sets ``last_seen_at``, and resets ``status``
    back to ``未解决`` when it had been marked 已解决/无需解决 (recurrence).
    """
    now = datetime.now()
    row = session.query(FaithCase).filter(FaithCase.eval_id == eval_id).first()
    if row is None:
        session.add(
            FaithCase(
                eval_id=eval_id,
                bucket=bucket,
                query=query,
                strategy=strategy,
                answer=answer,
                reason=reason,
                citations=citations,
                judge_model=judge_model,
                status="未解决",
                seen_count=1,
                first_seen_at=now,
                last_seen_at=now,
            )
        )
    else:
        row.answer = answer
        row.reason = reason
        row.citations = citations
        row.judge_model = judge_model
        row.seen_count = row.seen_count + 1
        row.last_seen_at = now
        if row.status in ("已解决", "无需解决"):
            row.status = "未解决"
    session.commit()
