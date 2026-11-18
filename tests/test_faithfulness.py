"""Tests for ch04 faithfulness judge + ``faith_cases`` upsert ledger."""

from datetime import datetime

from app.db.models import FaithCase
from evaluation.faithfulness import (
    FaithfulnessResult,
    judge_faithfulness,
    record_faith_case,
)


class FakeStructuredRunnable:
    """Callable stand-in coerced to a ``RunnableLambda`` by ``|``."""

    def __init__(self, result):
        self._result = result

    def __call__(self, inp):
        return self._result


class FakeLLM:
    """Records ``method`` and returns a runnable yielding a canned result."""

    def __init__(self, result):
        self._result = result
        self.called_model = None
        self.called_method = None

    def with_structured_output(self, model, method=None):
        self.called_model = model
        self.called_method = method
        return FakeStructuredRunnable(self._result)


def test_judge_faithfulness_passthrough_and_function_calling():
    llm = FakeLLM({"faithful": False, "reason": "声称退款24小时到账"})

    out = judge_faithfulness(
        llm,
        "退款什么时候到账",
        "一般24小时内到账",
        [{"n": 1, "chunk_id": 17}],
    )

    assert isinstance(out, FaithfulnessResult)
    assert out.faithful is False
    assert out.reason == "声称退款24小时到账"
    assert llm.called_model is FaithfulnessResult
    assert llm.called_method == "function_calling"


def test_record_faith_case_first_insert(session):
    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="七天无理由退货的条件是什么？",
        answer="支持七天无理由退货。",
        reason="声称退款到账时间",
        citations=[{"n": 1, "chunk_id": 28}],
        judge_model="qwen3:8b",
    )

    row = session.query(FaithCase).one()
    assert row.eval_id == "A01"
    assert row.bucket == "A_policy"
    assert row.strategy == "hybrid_rerank"
    assert row.query == "七天无理由退货的条件是什么？"
    assert row.answer == "支持七天无理由退货。"
    assert row.reason == "声称退款到账时间"
    assert row.citations == [{"n": 1, "chunk_id": 28}]
    assert row.judge_model == "qwen3:8b"
    assert row.status == "未解决"
    assert row.seen_count == 1
    assert row.first_seen_at is not None
    assert row.last_seen_at is not None


def test_record_faith_case_repeat_updates_seen_count_and_last_seen_at(session):
    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="q",
        answer="a1",
        reason="r1",
        citations=[{"n": 1}],
        judge_model="qwen3:8b",
    )
    row = session.query(FaithCase).one()
    assert row.seen_count == 1
    assert row.answer == "a1"

    old = datetime(2020, 1, 1)
    row.last_seen_at = old
    session.commit()

    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="q2",
        answer="a2",
        reason="r2",
        citations=[{"n": 2}],
        judge_model="qwen3:8b",
    )

    rows = session.query(FaithCase).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.seen_count == 2
    assert row.answer == "a2"
    assert row.reason == "r2"
    assert row.citations == [{"n": 2}]
    assert row.last_seen_at > old


def test_record_faith_case_recurrence_resets_status_from_resolved(session):
    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="q",
        answer="a",
        reason="r",
        citations=[{"n": 1}],
        judge_model="qwen3:8b",
    )
    row = session.query(FaithCase).one()
    row.status = "已解决"
    session.commit()

    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="q",
        answer="a",
        reason="r",
        citations=[{"n": 1}],
        judge_model="qwen3:8b",
    )

    row = session.query(FaithCase).one()
    assert row.status == "未解决"
    assert row.seen_count == 2


def test_record_faith_case_recurrence_resets_status_from_no_need(session):
    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="q",
        answer="a",
        reason="r",
        citations=[{"n": 1}],
        judge_model="qwen3:8b",
    )
    row = session.query(FaithCase).one()
    row.status = "无需解决"
    session.commit()

    record_faith_case(
        session,
        eval_id="A01",
        bucket="A_policy",
        query="q",
        answer="a",
        reason="r",
        citations=[{"n": 1}],
        judge_model="qwen3:8b",
    )

    row = session.query(FaithCase).one()
    assert row.status == "未解决"
    assert row.seen_count == 2
