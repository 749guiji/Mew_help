import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.db as app_db
import app.services.chat_service as chat_service_module
from app.db import Base
from app.db.models import Conversation, Message
from app.services.chat_service import ChatService
from app.tools.executor import ToolExecutor


class FakeLLM:
    """Stub LLM: ainvoke may return tool_calls; astream streams the final answer."""

    def __init__(self, reply="直接回答", tool_calls=None, chunks=None):
        self.reply = reply
        self.tool_calls = tool_calls
        self.chunks = chunks if chunks is not None else [reply]
        self.invokes = []
        self.streams = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.invokes.append([m.content for m in messages])
        if self.tool_calls is not None:
            return AIMessage(content="", tool_calls=self.tool_calls)
        return AIMessage(content=self.reply)

    async def astream(self, messages):
        self.streams.append([m.content for m in messages])
        for c in self.chunks:
            yield AIMessageChunk(content=c)


class _FakeTool:
    name = "query_faq"

    async def ainvoke(self, args):
        return "退货政策：7天无理由退货"


@pytest.fixture(autouse=True)
def sqlite_session_local(monkeypatch):
    """Swap the DB session for SQLite in-memory so no test touches real MySQL."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(app_db, "SessionLocal", SessionLocal)
    monkeypatch.setattr(chat_service_module, "SessionLocal", SessionLocal, raising=False)
    return SessionLocal


async def collect(agen):
    return [x async for x in agen]


def test_stream_chat_emits_events():
    fake = FakeLLM(reply="直接回答")
    svc = ChatService(fake)
    ev = asyncio.run(collect(svc.stream_chat(None, "你好")))
    assert ev[0]["type"] == "meta" and "session_id" in ev[0]
    assert [e["content"] for e in ev if e["type"] == "token"] == ["直接回答"]
    assert ev[-1]["type"] == "done"


def test_second_turn_receives_history():
    fake = FakeLLM(reply="收到")
    svc = ChatService(fake)
    e1 = asyncio.run(collect(svc.stream_chat(None, "我的订单123没发货")))
    sid = e1[0]["session_id"]
    asyncio.run(collect(svc.stream_chat(sid, "那帮我催一下")))
    second = fake.invokes[1]
    assert any("我的订单123没发货" in m for m in second)
    assert any("那帮我催一下" in m for m in second)


def test_unknown_session_id_starts_fresh():
    svc = ChatService(FakeLLM(reply="ok"))
    ev = asyncio.run(collect(svc.stream_chat("no-such-id", "hi")))
    assert ev[0]["type"] == "meta"
    assert ev[0]["session_id"] != "no-such-id"


def test_empty_reply_still_done():
    svc = ChatService(FakeLLM(reply=""))
    ev = asyncio.run(collect(svc.stream_chat(None, "hi")))
    assert ev[-1]["type"] == "done"


def test_stream_chat_emits_tool_frame_and_persists(sqlite_session_local):
    fake = FakeLLM(
        reply="根据FAQ，7天无理由退货。",
        tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}],
        chunks=["根据FAQ，", "7天无理由退货。"],
    )
    svc = ChatService(
        llm=fake,
        tools=[_FakeTool()],
        executor=ToolExecutor(registry={"query_faq": _FakeTool()}),
    )
    ev = asyncio.run(collect(svc.stream_chat(None, "怎么退货？")))

    assert [e for e in ev if e["type"] == "tool"] == [
        {"type": "tool", "name": "query_faq", "status": "running"},
        {"type": "tool", "name": "query_faq", "status": "done"},
    ]
    assert "".join(e["content"] for e in ev if e["type"] == "token") == "根据FAQ，7天无理由退货。"
    assert ev[-1]["type"] == "done"

    with sqlite_session_local() as s:
        assert s.query(Conversation).count() == 1
        rows = s.query(Message).order_by(Message.id).all()
        assert [m.role for m in rows] == ["user", "assistant", "tool", "assistant"]
        assert rows[0].content == "怎么退货？"
        assert rows[1].content is None
        saved_tc = rows[1].tool_calls[0]
        assert saved_tc["name"] == "query_faq"
        assert saved_tc["args"] == {"keyword": "退货"}
        assert saved_tc["id"] == "call_1"
        assert rows[2].tool_call_id == "call_1"
        assert rows[2].content == "退货政策：7天无理由退货"
        assert rows[3].content == "根据FAQ，7天无理由退货。"
