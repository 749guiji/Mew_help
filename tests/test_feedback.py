"""Tests for ch09 👎 feedback endpoint."""

import pytest

import app.api.routes.feedback as fb
import app.kb.retrieval as retrieval
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.db.models import LowConfidenceQuestion
from app.kb.hybrid import ScoredChunk
from app.main import create_app


class _FakeExtraction:
    async def extract(self, text):
        return None


@pytest.fixture
def sqlite_session_local(monkeypatch):
    # TestClient 起端点跑 worker 线程,SQLite 内存库默认每线程独立 → 用 StaticPool 共享单连接
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine)
    monkeypatch.setattr(fb, "SessionLocal", SL)
    return SL


def test_feedback_logs_pool_with_snapshot(monkeypatch, sqlite_session_local):
    chunk = ScoredChunk(
        chunk_id=1, score=0.9, section_path="运费说明", category="运费",
        content_type="policy", answer="满99包邮", questions="",
    )
    monkeypatch.setattr(retrieval, "default_retrieve", lambda q: [chunk])

    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post("/api/feedback", json={"conversation_id": None, "message": "运费怎么算"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}

    with sqlite_session_local() as s:
        rows = s.query(LowConfidenceQuestion).all()
        assert len(rows) == 1
        assert rows[0].source == "user_feedback"
        assert rows[0].retrieved_chunks[0]["score"] == 0.9


def test_feedback_retrieval_failure_snapshot_none(monkeypatch, sqlite_session_local):
    monkeypatch.setattr(
        retrieval, "default_retrieve", lambda q: (_ for _ in ()).throw(RuntimeError("down"))
    )

    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post("/api/feedback", json={"conversation_id": 3, "message": "运费怎么算"})
    assert r.status_code == 200

    with sqlite_session_local() as s:
        row = s.query(LowConfidenceQuestion).one()
        assert row.source == "user_feedback"
        assert row.retrieved_chunks is None
        assert row.conversation_id == 3
