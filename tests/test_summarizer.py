"""Tests for the ch07 background summary task."""

import time
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.core.summarizer as summarizer_module
from app.db import Base
from app.db.models import Conversation, ConversationSummary, Message


class FakeLLM:
    def __init__(self, content):
        self._content = content

    def __call__(self, inp):
        return SimpleNamespace(content=self._content)


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(summarizer_module, "SessionLocal", SessionLocal)
    return SessionLocal


def test_summarize_layer2_appends_segment_and_moves_anchor(sqlite_session_local):
    with sqlite_session_local() as s:
        conv = Conversation(user_id="u1", status="进行中", summary_upto_msg_id=0, layer1_from_msg_id=2)
        s.add(conv)
        s.commit()
        s.refresh(conv)
        cid = conv.id
        s.add(Message(conversation_id=cid, role="user", content="订单1001能退吗"))
        s.add(Message(conversation_id=cid, role="assistant", content="我看看"))
        s.commit()

    summarizer_module.summarize_layer2(cid, llm=FakeLLM("用户问过订单1001想退货,还没解决"))

    with sqlite_session_local() as s:
        segs = s.query(ConversationSummary).order_by(ConversationSummary.seq).all()
        assert len(segs) == 1 and segs[0].seq == 1
        assert segs[0].content == "用户问过订单1001想退货,还没解决"
        conv = s.get(Conversation, cid)
        assert "订单1001" in conv.summary
        assert conv.summary_upto_msg_id == 2  # 追到 layer1_from


def test_trigger_only_when_over_budget(monkeypatch, sqlite_session_local):
    calls = []
    monkeypatch.setattr(summarizer_module, "summarize_layer2", lambda cid, **kw: calls.append(cid))

    summarizer_module.trigger_summary_if_needed(1, 100, 200)  # 未超 → 不触发
    assert calls == []

    summarizer_module.trigger_summary_if_needed(1, 300, 200)  # 超 → 后台触发
    time.sleep(0.2)
    assert calls == [1]
