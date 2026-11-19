"""Tests for ch04 generation quality helpers (self-check / pool / citations)."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.db.models import LowConfidenceQuestion
from app.kb.hybrid import ScoredChunk
from app.services.generation_quality import (
    SelfCheckResult,
    build_citations,
    format_evidence,
    log_low_confidence,
    reorder_for_generation,
    self_check,
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


def _make_session_local():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _chunk(chunk_id, score, section_path, category, answer):
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        section_path=section_path,
        category=category,
        content_type="faq",
        answer=answer,
        questions="",
    )


def test_self_check_passthrough_and_function_calling():
    llm = FakeLLM(SelfCheckResult(sufficient=False, reason="缺运费"))

    out = self_check(llm, "退货运费谁出", "证据内容")

    assert isinstance(out, SelfCheckResult)
    assert out.sufficient is False
    assert out.reason == "缺运费"
    assert llm.called_model is SelfCheckResult
    assert llm.called_method == "function_calling"


def test_self_check_falls_back_when_chain_returns_none():
    llm = FakeLLM(None)

    out = self_check(llm, "退货运费谁出", "证据内容")

    assert isinstance(out, SelfCheckResult)
    assert out.sufficient is True
    assert out.reason == ""
    assert llm.called_model is SelfCheckResult
    assert llm.called_method == "function_calling"


def test_log_low_confidence_writes_row():
    session_local = _make_session_local()

    with session_local() as s:
        log_low_confidence(
            s,
            conversation_id=7,
            raw_question="退货运费谁出",
            source="self_check",
            reason="缺运费",
        )

    with session_local() as s:
        rows = s.query(LowConfidenceQuestion).all()
        assert len(rows) == 1
        assert rows[0].source == "self_check"
        assert rows[0].reason == "缺运费"
        assert rows[0].raw_question == "退货运费谁出"
        assert rows[0].conversation_id == 7


def test_format_evidence_numbers_and_uses_section_path():
    chunks = [
        _chunk(1, 0.9, "售后退款", "退款", "7天无理由"),
        _chunk(2, 0.8, None, "运费", "满 99 包邮"),
    ]

    out = format_evidence(chunks)

    assert out == "[1] 售后退款: 7天无理由\n[2] 运费: 满 99 包邮"


def test_build_citations_shape():
    chunks = [_chunk(5, 0.9, "售后退款", "退款", "x" * 250)]

    out = build_citations(chunks)

    assert out == [
        {
            "n": 1,
            "chunk_id": 5,
            "section_path": "售后退款",
            "answer_snippet": "x" * 200,
        }
    ]


def test_reorder_places_top_two_at_ends():
    chunks = [_chunk(i, 1.0 - i * 0.1, f"s{i}", "退款", f"答案{i}") for i in range(1, 6)]

    ordered = reorder_for_generation(chunks)

    assert [c.chunk_id for c in ordered] == [1, 3, 5, 4, 2]


def test_reorder_leaves_two_or_fewer_unchanged():
    assert reorder_for_generation([]) == []

    one = [_chunk(1, 0.9, "s1", "退款", "a")]
    assert [c.chunk_id for c in reorder_for_generation(one)] == [1]

    two = [_chunk(1, 0.9, "s1", "退款", "a"), _chunk(2, 0.8, "s2", "退款", "b")]
    assert [c.chunk_id for c in reorder_for_generation(two)] == [1, 2]


def test_format_evidence_and_citations_share_numbering():
    chunks = [
        _chunk(1, 0.9, "s1", "退款", "答案1"),
        _chunk(2, 0.8, "s2", "退款", "答案2"),
        _chunk(3, 0.7, "s3", "退款", "答案3"),
    ]

    evidence = format_evidence(chunks)
    citations = build_citations(chunks)

    # Reorder → [chunk 1, chunk 3, chunk 2]; citation n must follow the same order.
    assert evidence.splitlines()[0].startswith("[1]")
    assert [c["chunk_id"] for c in citations] == [1, 3, 2]
    assert [c["n"] for c in citations] == [1, 2, 3]
