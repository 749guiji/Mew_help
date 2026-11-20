"""Tests for the ch07 degradation wiring (layer1 anchor move + summary trigger)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.core.degradation as degradation_module
from app.core.config import Settings
from app.db import Base
from app.db.models import Conversation, Message


def _demo_settings():
    return Settings(
        _env_file=None,
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(degradation_module, "SessionLocal", SessionLocal)
    monkeypatch.setattr(degradation_module, "get_settings", _demo_settings)
    return SessionLocal


def test_apply_degradation_moves_layer1_anchor(sqlite_session_local, monkeypatch):
    monkeypatch.setattr(degradation_module, "trigger_summary_if_needed", lambda *a, **k: None)
    with sqlite_session_local() as s:
        conv = Conversation(user_id="u1", status="进行中")
        s.add(conv)
        s.commit()
        s.refresh(conv)
        cid = conv.id
        for i in range(5):
            s.add(Message(conversation_id=cid, role="user", content="x" * 1000))
        s.commit()

    # 层1 用量 5000 > 层1 预算 3954 → 上移锚点
    degradation_module.apply_degradation(cid, layer1_chars=5000, layer2_chars=0)

    with sqlite_session_local() as s:
        conv = s.get(Conversation, cid)
        assert conv.layer1_from_msg_id is not None  # 锚点被写


def test_apply_degradation_triggers_summary(sqlite_session_local, monkeypatch):
    calls = []
    monkeypatch.setattr(
        degradation_module,
        "trigger_summary_if_needed",
        lambda cid, u, b: calls.append((cid, u, b)),
    )
    with sqlite_session_local() as s:
        conv = Conversation(user_id="u1", status="进行中")
        s.add(conv)
        s.commit()
        cid = conv.id

    # 层2 用量 2000 > 层2 预算 1695 → 触发摘要
    degradation_module.apply_degradation(cid, layer1_chars=0, layer2_chars=2000)
    assert calls == [(cid, 2000, 1695)]
