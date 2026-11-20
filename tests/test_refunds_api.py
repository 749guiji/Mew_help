"""Tests for the POST /api/refunds endpoint."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.routes.refunds as refunds_module
from app.db import Base
from app.db.models import Conversation, RefundRequest
from app.main import create_app


@pytest.fixture
def sqlite_session_local(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'refunds.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(refunds_module, "SessionLocal", SessionLocal)
    return SessionLocal


def _client():
    return TestClient(create_app())


def test_create_refund_writes_row(sqlite_session_local):
    with sqlite_session_local() as s:
        conv = Conversation(user_id="sid1", status="进行中")
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = conv.id

    r = _client().post(
        "/api/refunds",
        json={"session_id": "sid1", "order_id": "1001", "reason": "7天无理由"},
    )
    assert r.status_code == 200
    assert r.json()["refund_id"]

    with sqlite_session_local() as s:
        ref = s.query(RefundRequest).one()
        assert ref.conversation_id == conv_id
        assert ref.order_id == "1001"
        assert ref.reason == "7天无理由"
        assert ref.status == "待处理"


def test_refunds_rejects_invalid_reason(sqlite_session_local):
    r = _client().post(
        "/api/refunds",
        json={"session_id": "sid1", "order_id": "1001", "reason": "随便写"},
    )
    assert r.status_code == 422


def test_refunds_rejects_blank_order_id(sqlite_session_local):
    r = _client().post(
        "/api/refunds",
        json={"session_id": "sid1", "order_id": "   ", "reason": "7天无理由"},
    )
    assert r.status_code == 422
