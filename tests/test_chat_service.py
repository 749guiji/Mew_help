import asyncio
from langchain_core.messages import AIMessageChunk
from app.services.chat_service import ChatService

class FakeLLM:
    def __init__(self, chunks): self.chunks = chunks; self.calls = []
    async def astream(self, messages):
        self.calls.append([m.content for m in messages])
        for c in self.chunks:
            yield AIMessageChunk(content=c)

async def collect(agen):
    return [x async for x in agen]

def test_stream_chat_emits_events():
    fake = FakeLLM(["你", "好"])
    svc = ChatService(fake)
    ev = asyncio.run(collect(svc.stream_chat(None, "你好")))
    assert ev[0]["type"] == "meta" and "session_id" in ev[0]
    assert [e["content"] for e in ev if e["type"] == "token"] == ["你", "好"]
    assert ev[-1]["type"] == "done"

def test_second_turn_receives_history():
    fake = FakeLLM(["收到"])
    svc = ChatService(fake)
    e1 = asyncio.run(collect(svc.stream_chat(None, "我的订单123没发货")))
    sid = e1[0]["session_id"]
    asyncio.run(collect(svc.stream_chat(sid, "那帮我催一下")))
    second = fake.calls[1]
    assert any("我的订单123没发货" in m for m in second)
    assert any("那帮我催一下" in m for m in second)

def test_unknown_session_id_starts_fresh():
    svc = ChatService(FakeLLM(["ok"]))
    ev = asyncio.run(collect(svc.stream_chat("no-such-id", "hi")))
    assert ev[0]["type"] == "meta"
    assert ev[0]["session_id"] != "no-such-id"

def test_empty_stream_still_done():
    svc = ChatService(FakeLLM([]))
    ev = asyncio.run(collect(svc.stream_chat(None, "hi")))
    assert ev[-1]["type"] == "done"
