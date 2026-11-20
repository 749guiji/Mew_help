"""Tests for the conversation sidebar read-only endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.api.routes.conversations as conv_routes
from app.db import Base
from app.db.models import Conversation, Message
from app.main import create_app


class _FakeExtraction:
    async def extract(self, text):
        return None


@pytest.fixture
def sqlite_session_local(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'conv.db'}")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(conv_routes, "SessionLocal", SessionLocal)
    return SessionLocal


def test_list_and_messages(sqlite_session_local):
    with sqlite_session_local() as s:
        c1 = Conversation(user_id="u1", status="进行中", summary="有摘要")
        c2 = Conversation(user_id="u1", status="进行中")
        s.add_all([c1, c2])
        s.commit()
        s.refresh(c1)
        s.refresh(c2)
        s.add(Message(conversation_id=c1.id, role="user", content="订单1001能退吗"))
        s.add(Message(conversation_id=c1.id, role="assistant", content="可以退"))
        s.commit()
        c1_id, c2_id = c1.id, c2.id

    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.get("/api/conversations", params={"user_id": "u1"})
        assert r.status_code == 200
        convs = r.json()["conversations"]
        assert len(convs) == 2
        assert convs[0]["id"] == c2_id  # 新在前
        summarized = {x["id"]: x["summarized"] for x in convs}
        assert summarized[c1_id] is True and summarized[c2_id] is False

        r2 = c.get(f"/api/conversations/{c1_id}/messages")
        msgs = r2.json()["messages"]
        assert [m["content"] for m in msgs] == ["订单1001能退吗", "可以退"]
