import json
from fastapi.testclient import TestClient
import app.main as main_module
from app.main import create_app
from app.schemas.extraction import AfterSalesExtraction

class FakeChatService:
    def __init__(self): self.last = None
    async def stream_chat(self, session_id, message):
        self.last = (session_id, message)
        yield {"type": "meta", "session_id": session_id or "s1"}
        yield {"type": "token", "content": "你好"}
        yield {"type": "done"}

class FakeToolChatService:
    def __init__(self): self.last = None
    async def stream_chat(self, session_id, message):
        self.last = (session_id, message)
        yield {"type": "tool", "name": "query_faq", "status": "running"}
        yield {"type": "done"}

class FakeExtractionService:
    async def extract(self, text):
        return AfterSalesExtraction(order_id="123", request_type="退货", expected_resolution="退款")

def test_chat_sse_stream():
    app = create_app(chat_service=FakeChatService())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "你好"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert '"type":"token"' in r.text and '"type":"done"' in r.text

def test_chat_passes_session_id():
    fake = FakeChatService()
    app = create_app(chat_service=fake)
    with TestClient(app) as c:
        c.post("/chat", json={"session_id": "abc", "message": "hi"})
    assert fake.last[0] == "abc"

def test_chat_empty_message_422():
    app = create_app(chat_service=FakeChatService())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "   "})
    assert r.status_code == 422

def test_chat_sse_includes_tool_frame():
    app = create_app(chat_service=FakeToolChatService())
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "怎么退货"})
    assert r.status_code == 200
    assert '"type":"tool"' in r.text

def test_create_app_default_wires_chat_service(monkeypatch):
    captured = {}

    def fake_chat_service():
        captured["called"] = True
        return FakeChatService()

    monkeypatch.setattr(main_module, "ChatService", fake_chat_service)
    create_app(extraction_service=FakeExtractionService())
    assert captured["called"] is True

def test_extract_json():
    app = create_app(extraction_service=FakeExtractionService())
    with TestClient(app) as c:
        r = c.post("/extract", json={"text": "订单123退货"})
    assert r.status_code == 200
    assert r.json() == {"order_id": "123", "request_type": "退货", "expected_resolution": "退款"}
