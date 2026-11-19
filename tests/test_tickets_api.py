"""Tests for the POST /api/tickets endpoint."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.routes.tickets as tickets_module
from app.db import Base
from app.db.models import Conversation, Ticket
from app.main import create_app


@pytest.fixture
def sqlite_session_local(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'tickets.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(tickets_module, "SessionLocal", SessionLocal)
    return SessionLocal


def _client():
    return TestClient(create_app())


def test_create_ticket_writes_row(sqlite_session_local):
    with sqlite_session_local() as s:
        conv = Conversation(user_id="sid1", status="进行中")
        s.add(conv)
        s.commit()
        s.refresh(conv)
        conv_id = conv.id

    r = _client().post(
        "/api/tickets",
        json={"session_id": "sid1", "description": "物流太慢", "ticket_type": "投诉"},
    )
    assert r.status_code == 200

    with sqlite_session_local() as s:
        t = s.query(Ticket).one()
        assert t.conversation_id == conv_id
        assert t.description == "物流太慢"
        assert t.ticket_type == "投诉"
        assert t.status == "待处理"


def test_tickets_rejects_blank_description(sqlite_session_local):
    r = _client().post(
        "/api/tickets",
        json={"session_id": "sid1", "description": "   ", "ticket_type": "投诉"},
    )
    assert r.status_code == 422
