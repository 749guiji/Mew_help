"""Tests for the ch05 graph-backed chat service."""

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db as app_db
import app.services.chat_service as chat_service_module
from app.db import Base
from app.db.models import Conversation, Message
from app.kb.hybrid import ScoredChunk
from app.services.chat_service import ChatService


class FakeGraph:
    """Stateful fake: accumulates messages across turns (like a checkpointer)."""

    def __init__(self, result=None):
        self.result = result or {}
        self.inputs = []
        self._messages = []

    async def aget_state(self, config=None):
        class _Snapshot:
            def __init__(self, values):
                self.values = values

        return _Snapshot({"messages": list(self._messages)})

    async def ainvoke(self, input, config=None):
        self.inputs.append((input, config))
        for m in input.get("messages", []):
            if isinstance(m, tuple):
                self._messages.append(HumanMessage(content=m[1]))
            else:
                self._messages.append(m)
        for m in self.result.get("messages", []):
            self._messages.append(m)
        state = dict(self.result)
        state["messages"] = list(self._messages)
        return state


@pytest.fixture(autouse=True)
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(app_db, "SessionLocal", SessionLocal)
    monkeypatch.setattr(chat_service_module, "SessionLocal", SessionLocal, raising=False)
    return SessionLocal


async def collect(agen):
    return [x async for x in agen]


def _chunk(chunk_id=1, section_path="运费说明", answer="满99包邮"):
    return ScoredChunk(
        chunk_id=chunk_id,
        score=0.9,
        section_path=section_path,
        category="运费",
        content_type="policy",
        answer=answer,
        questions="运费怎么算",
    )


def test_stream_emits_meta_token_done():
    graph = FakeGraph({"messages": [AIMessage(content="直接回答")]})
    svc = ChatService(graph=graph)
    ev = asyncio.run(collect(svc.stream_chat(None, "你好")))
    assert ev[0]["type"] == "meta" and "session_id" in ev[0]
    assert [e["content"] for e in ev if e["type"] == "token"] == ["直接回答"]
    assert ev[-1]["type"] == "done"


def test_complaint_emits_handoff(sqlite_session_local):
    graph = FakeGraph({
        "messages": [AIMessage(content="非常抱歉给您带来了不好的体验。")],
        "handoff_suggestions": ["转人工", "建工单"],
    })
    svc = ChatService(graph=graph)
    ev = asyncio.run(collect(svc.stream_chat(None, "我要投诉")))
    handoff = [e for e in ev if e["type"] == "handoff"]
    assert handoff == [{"type": "handoff", "suggestions": ["转人工", "建工单"]}]
    assert [e["content"] for e in ev if e["type"] == "token"] == ["非常抱歉给您带来了不好的体验。"]
    assert ev[-1]["type"] == "done"


def test_knowledge_emits_citations(sqlite_session_local):
    graph = FakeGraph({
        "messages": [AIMessage(content="满99包邮")],
        "retrieval_chunks": [_chunk()],
    })
    svc = ChatService(graph=graph)
    ev = asyncio.run(collect(svc.stream_chat(None, "运费怎么算")))
    cit = [e for e in ev if e["type"] == "citations"]
    assert len(cit) == 1
    assert cit[0]["citations"][0]["section_path"] == "运费说明"
    assert [e["content"] for e in ev if e["type"] == "token"] == ["满99包邮"]


def test_business_emits_tool_frames(sqlite_session_local):
    graph = FakeGraph({
        "messages": [
            HumanMessage(content="订单 1001 物流"),
            AIMessage(content="", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]),
            ToolMessage(content="派送中", tool_call_id="c1"),
            AIMessage(content="订单在派送中"),
        ],
    })
    svc = ChatService(graph=graph)
    ev = asyncio.run(collect(svc.stream_chat(None, "订单 1001 物流")))
    tools = [e for e in ev if e["type"] == "tool"]
    assert tools == [{"type": "tool", "name": "query_logistics", "status": "done"}]
    assert [e["content"] for e in ev if e["type"] == "token"] == ["订单在派送中"]


def test_second_turn_uses_same_thread_id(sqlite_session_local):
    graph = FakeGraph({"messages": [AIMessage(content="收到")]})
    svc = ChatService(graph=graph)
    e1 = asyncio.run(collect(svc.stream_chat(None, "第一句")))
    sid = e1[0]["session_id"]
    asyncio.run(collect(svc.stream_chat(sid, "第二句")))
    # 两次都用同一个 thread_id(会话记忆)
    assert graph.inputs[0][1]["configurable"]["thread_id"] == sid
    assert graph.inputs[1][1]["configurable"]["thread_id"] == sid


def test_unknown_session_id_starts_fresh():
    graph = FakeGraph({"messages": [AIMessage(content="ok")]})
    svc = ChatService(graph=graph)
    ev = asyncio.run(collect(svc.stream_chat("no-such-id", "hi")))
    assert ev[0]["session_id"] != "no-such-id"


def test_db_failure_is_logged_and_swallowed(monkeypatch, caplog):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(chat_service_module, "SessionLocal", boom)
    graph = FakeGraph({"messages": [AIMessage(content="ok")]})
    svc = ChatService(graph=graph)
    with caplog.at_level(logging.WARNING, logger="app.services.chat_service"):
        ev = asyncio.run(collect(svc.stream_chat(None, "hi")))
    assert ev[-1]["type"] == "done"
    assert "db down" in caplog.text


def test_persists_user_and_assistant(sqlite_session_local):
    graph = FakeGraph({"messages": [AIMessage(content="答案")]})
    svc = ChatService(graph=graph)
    asyncio.run(collect(svc.stream_chat(None, "问题")))
    with sqlite_session_local() as s:
        assert s.query(Conversation).count() == 1
        rows = s.query(Message).order_by(Message.id).all()
        assert [(m.role, m.content) for m in rows] == [("user", "问题"), ("assistant", "答案")]


def test_second_turn_has_no_stale_handoff_or_citations(sqlite_session_local):
    graph = FakeGraph()
    graph.result = {"messages": [AIMessage(content="安抚")], "handoff_suggestions": ["转人工", "建工单"]}
    svc = ChatService(graph=graph)
    e1 = asyncio.run(collect(svc.stream_chat(None, "我要投诉")))
    assert [e for e in e1 if e["type"] == "handoff"]
    sid = e1[0]["session_id"]

    graph.result = {"messages": [AIMessage(content="您好呀")]}
    e2 = asyncio.run(collect(svc.stream_chat(sid, "你好")))

    assert not [e for e in e2 if e["type"] == "handoff"]
    assert not [e for e in e2 if e["type"] == "citations"]
    assert [e["content"] for e in e2 if e["type"] == "token"] == ["您好呀"]
