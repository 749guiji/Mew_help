"""Tests for POST /api/actions/resume and /api/actions/create-refund."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.routes.actions as actions_module
from app.db import Base
from app.db.models import Conversation, Ticket
from app.main import create_app


class _FakeExtraction:
    async def extract(self, text):
        return None


@pytest.fixture
def sqlite_session_local(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'actions.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(actions_module, "SessionLocal", SessionLocal)
    return SessionLocal


class FakeChatService:
    def __init__(self):
        self.last = None

    async def stream_resume(self, session_id, order_id):
        self.last = (session_id, order_id)
        yield {"type": "token", "content": "这单可以退"}
        yield {"type": "done"}


def test_resume_endpoint_streams():
    fake = FakeChatService()
    app = create_app(chat_service=fake, extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post("/api/actions/resume", json={"conversation_id": 3, "order_id": "1001"})
    assert r.status_code == 200
    assert "这单可以退" in r.text
    assert fake.last == (3, "1001")


def test_resume_endpoint_confirm():
    fake = FakeChatService()
    app = create_app(chat_service=fake, extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post("/api/actions/resume", json={"conversation_id": 3, "confirm": True})
    assert r.status_code == 200
    assert "这单可以退" in r.text
    assert fake.last == (3, {"confirm": True})


def test_create_refund_writes_ticket(sqlite_session_local):
    with sqlite_session_local() as s:
        conv = Conversation(user_id="sid1", status="进行中")
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = conv.id

    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post(
            "/api/actions/create-refund",
            json={"session_id": "sid1", "order_id": "1001", "reason": "质量问题"},
        )
    assert r.status_code == 200
    assert r.json()["ticket_no"]

    with sqlite_session_local() as s:
        t = s.query(Ticket).filter(Ticket.ticket_type == "退款").one()
        assert t.conversation_id == conv_id
        assert "1001" in t.description and "质量问题" in t.description


def test_create_refund_rejects_bad_reason(sqlite_session_local):
    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.post(
            "/api/actions/create-refund",
            json={"session_id": "sid1", "order_id": "1001", "reason": "乱填"},
        )
    assert r.status_code == 422
