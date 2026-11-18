import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    FaithCase,
    KnowledgeChunk,
    LowConfidenceQuestion,
    QaExtractionStaging,
)


def test_low_confidence_question_columns():
    cols = {c.name for c in LowConfidenceQuestion.__table__.columns}
    assert cols == {
        "id",
        "conversation_id",
        "raw_question",
        "source",
        "reason",
        "created_at",
    }


def test_low_confidence_question_source_enum_values():
    source = LowConfidenceQuestion.__table__.columns["source"].type
    assert source.enums == ["retrieval_low_conf", "self_check", "user_feedback"]


def test_low_confidence_question_roundtrip(session):
    q = LowConfidenceQuestion(
        conversation_id=1,
        raw_question="运费怎么算",
        source="retrieval_low_conf",
        reason="证据不足",
    )
    session.add(q)
    session.commit()
    got = session.get(LowConfidenceQuestion, q.id)
    assert got.raw_question == "运费怎么算"
    assert got.source == "retrieval_low_conf"


def test_faith_case_columns():
    cols = {c.name for c in FaithCase.__table__.columns}
    assert cols == {
        "id",
        "eval_id",
        "bucket",
        "query",
        "strategy",
        "answer",
        "reason",
        "citations",
        "judge_model",
        "status",
        "seen_count",
        "first_seen_at",
        "last_seen_at",
        "resolution",
        "resolved_at",
    }


def test_faith_case_status_enum_values():
    status = FaithCase.__table__.columns["status"].type
    assert status.enums == ["未解决", "已解决", "无需解决"]


def test_faith_case_roundtrip_defaults(session):
    f = FaithCase(
        eval_id="A43",
        bucket="A_policy",
        query="运费怎么算",
        answer="满99包邮",
        reason="证据未提及免邮规则",
    )
    session.add(f)
    session.commit()
    got = session.get(FaithCase, f.id)
    assert got.status == "未解决"
    assert got.strategy == "hybrid_rerank"
    assert got.seen_count == 1


def test_faith_case_eval_id_unique(session):
    session.add(
        FaithCase(
            eval_id="A43",
            bucket="A_policy",
            query="q",
            answer="a",
            reason="r",
        )
    )
    session.commit()
    session.add(
        FaithCase(
            eval_id="A43",
            bucket="A_policy",
            query="q",
            answer="a",
            reason="r",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_knowledge_chunk_roundtrip(session):
    c = KnowledgeChunk(category="售后>运费", questions="运费怎么收", answer="满99包邮")
    session.add(c); session.commit()
    assert c.vectorize_status == "pending" and c.vector_id is None
    # 自引用指针:插两块后回填
    c2 = KnowledgeChunk(category="售后", questions="退货", answer="7天无理由")
    session.add(c2); session.commit()
    c.next_chunk_id = c2.id; c2.prev_chunk_id = c.id; session.commit()
    assert session.get(KnowledgeChunk, c.id).next_chunk_id == c2.id


def test_qa_staging_defaults(session):
    s = QaExtractionStaging(batch_no="b1", question="q", answer="a")
    session.add(s); session.commit()
    assert s.status == "extracted"
