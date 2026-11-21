"""Tests for ch09 review-queue admin API."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.routes.review as review
from app.db import Base
from app.db.models import KnowledgeChunk, ReviewQueue
from app.main import create_app


class _FakeExtraction:
    async def extract(self, text):
        return None


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine)
    monkeypatch.setattr(review, "SessionLocal", SL)
    return SL


def test_approve_writes_chunk_and_status(sqlite_session_local, monkeypatch):
    monkeypatch.setattr(review, "_vectorize_new_chunk", lambda session: None)

    with sqlite_session_local() as s:
        rq = ReviewQueue(normalized_question="运费怎么算", occurrence_count=2)
        s.add(rq)
        s.commit()
        s.refresh(rq)
        rid = rq.id

    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post(f"/api/review/{rid}/approve", json={"approved_answer": "满99包邮"})
    assert r.status_code == 200

    with sqlite_session_local() as s:
        rq = s.get(ReviewQueue, rid)
        assert rq.review_status == "通过"
        assert rq.approved_answer == "满99包邮"
        chunks = s.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "faq").all()
        assert len(chunks) == 1
        assert chunks[0].answer == "满99包邮"


def test_approve_vectorize_failure_keeps_pending(sqlite_session_local, monkeypatch):
    def boom(session):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(review, "_vectorize_new_chunk", boom)

    with sqlite_session_local() as s:
        rq = ReviewQueue(normalized_question="退货政策", occurrence_count=1)
        s.add(rq)
        s.commit()
        s.refresh(rq)
        rid = rq.id

    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post(f"/api/review/{rid}/approve", json={"approved_answer": "7天无理由"})
    assert r.status_code == 200

    with sqlite_session_local() as s:
        rq = s.get(ReviewQueue, rid)
        assert rq.review_status == "通过"  # 状态仍通过
        chunk = s.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "faq").one()
        assert chunk.vectorize_status == "pending"  # 向量化失败留 pending,不丢数据
