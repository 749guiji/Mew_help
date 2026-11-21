"""Tests for ch10 batch topic-classification job."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.jobs.classify_topics as ct
from app.db import Base
from app.db.models import LowConfidenceQuestion, TopicClassification


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine)
    monkeypatch.setattr(ct, "SessionLocal", SL)
    return SL


class _FakeResp:
    def __init__(self, labels):
        self._labels = labels

    def raise_for_status(self):
        pass

    def json(self):
        return {"labels": self._labels}


def test_classify_writes_rows(monkeypatch, sqlite_session_local):
    with sqlite_session_local() as s:
        s.add_all([
            LowConfidenceQuestion(raw_question="买大了想退", source="user_feedback"),
            LowConfidenceQuestion(raw_question="快递到哪了", source="retrieval_low_conf"),
        ])
        s.commit()

    monkeypatch.setattr(
        ct.httpx, "post",
        lambda *a, **k: _FakeResp([["尺码", "退换货"], ["物流"]]),
    )
    result = ct.classify_topics()
    assert result["processed"] == 2

    with sqlite_session_local() as s:
        rows = s.query(TopicClassification).all()
        assert len(rows) == 2
        assert rows[0].labels == ["尺码", "退换货"]


def test_classify_service_down_no_crash(monkeypatch, sqlite_session_local):
    with sqlite_session_local() as s:
        s.add(LowConfidenceQuestion(raw_question="x", source="user_feedback"))
        s.commit()

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(ct.httpx, "post", boom)
    result = ct.classify_topics()
    assert result["processed"] == 0

    with sqlite_session_local() as s:
        assert s.query(TopicClassification).count() == 0
