"""Tests for ch09 flywheel pipeline (normalize + dedupe + enqueue)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.jobs.flywheel as fw
from app.db import Base
from app.db.models import LowConfidenceQuestion, ReviewQueue


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine)
    monkeypatch.setattr(fw, "SessionLocal", SL)
    return SL


def test_new_gap_creates_review_row(monkeypatch, sqlite_session_local):
    with sqlite_session_local() as s:
        q = LowConfidenceQuestion(raw_question="运费咋算", source="retrieval_low_conf")
        s.add(q)
        s.commit()
        s.refresh(q)
        qid = q.id

    monkeypatch.setattr(fw, "normalize_question", lambda llm, raw: ("运费怎么算", "满99包邮"))
    monkeypatch.setattr(fw, "dedupe_question", lambda llm, norm, cands: None)

    result = fw.run_flywheel()
    assert result["new"] == 1 and result["processed"] == 1 and result["merged"] == 0

    with sqlite_session_local() as s:
        rq = s.query(ReviewQueue).one()
        assert rq.normalized_question == "运费怎么算"
        assert rq.occurrence_count == 1
        q = s.get(LowConfidenceQuestion, qid)
        assert q.matched_review_id == rq.id


def test_duplicate_merges_occurrence(monkeypatch, sqlite_session_local):
    with sqlite_session_local() as s:
        rq = ReviewQueue(normalized_question="运费怎么算", occurrence_count=1)
        s.add(rq)
        s.commit()
        s.refresh(rq)
        rid = rq.id
        q = LowConfidenceQuestion(raw_question="邮费多少", source="user_feedback")
        s.add(q)
        s.commit()
        s.refresh(q)
        qid = q.id

    monkeypatch.setattr(fw, "normalize_question", lambda llm, raw: ("运费怎么算", None))
    monkeypatch.setattr(fw, "dedupe_question", lambda llm, norm, cands: rid)

    result = fw.run_flywheel()
    assert result["merged"] == 1 and result["new"] == 0

    with sqlite_session_local() as s:
        assert s.query(ReviewQueue).count() == 1
        rq = s.get(ReviewQueue, rid)
        assert rq.occurrence_count == 2
        q = s.get(LowConfidenceQuestion, qid)
        assert q.matched_review_id == rid
