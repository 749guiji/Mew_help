"""Tests for the chapter 3 tool functions and registry."""

import json
import re

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.tools.db_tools as db_tools
from app.db import Base
from app.db.models import Ticket
from app.db.seed import seed_faq
from app.kb.hybrid import ScoredChunk
from app.tools import registry
from app.tools.context import current_conversation_id


def _invoke(name, args):
    """调用注册中心里某个内置工具(旧 TOOL_REGISTRY 的等价物)。"""
    return registry.get(name).impl.invoke(args)


@pytest.fixture
def sqlite_session_local(monkeypatch):
    """Swap app.tools.db_tools.SessionLocal for a SQLite in-memory sessionmaker."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(db_tools, "SessionLocal", SessionLocal)
    with SessionLocal() as s:
        seed_faq(s)
    return SessionLocal


def test_registry_has_five_builtin_tools():
    names = {t.name for t in registry.builtin()}
    assert names == {
        "query_order",
        "query_product",
        "query_faq",
        "create_ticket",
        "submit_refund",
    }
    # query_logistics 已下线(由物流 MCP Server 接管)


def test_query_order_returns_str():
    out = _invoke("query_order", {"order_id": "1001"})
    assert isinstance(out, str) and "1001" in out


def test_query_order_deterministic():
    a = _invoke("query_order", {"order_id": "1001"})
    b = _invoke("query_order", {"order_id": "1001"})
    assert a == b


def test_query_product_returns_str():
    out = _invoke("query_product", {"product_name": "保温杯"})
    assert isinstance(out, str) and "保温杯" in out


def _chunk(chunk_id, score, section_path, category, answer):
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        section_path=section_path,
        category=category,
        content_type="faq",
        answer=answer,
        questions="",
    )


def test_query_faq_empty_retrieval_refuses(monkeypatch):
    monkeypatch.setattr(db_tools, "default_retrieve", lambda q: [])
    out = json.loads(_invoke("query_faq", {"keyword": "邮费"}))
    assert out == {"refused": True, "evidence": "", "citations": []}


def test_query_faq_none_retrieval_refuses(monkeypatch):
    monkeypatch.setattr(db_tools, "default_retrieve", lambda q: None)
    out = json.loads(_invoke("query_faq", {"keyword": "邮费"}))
    assert out == {"refused": True, "evidence": "", "citations": []}


def test_query_faq_low_score_refuses(monkeypatch):
    monkeypatch.setattr(
        db_tools,
        "default_retrieve",
        lambda q: [_chunk(1, 0.1, "运费", "运费", "包邮")],
    )
    out = json.loads(_invoke("query_faq", {"keyword": "邮费"}))
    assert out == {"refused": True, "evidence": "", "citations": []}


def test_query_faq_match_returns_json_contract(monkeypatch):
    monkeypatch.setattr(
        db_tools,
        "default_retrieve",
        lambda q: [_chunk(1, 0.9, "售后退款", "退款", "7天无理由退货")],
    )
    out = json.loads(_invoke("query_faq", {"keyword": "退货"}))

    assert out["refused"] is False
    assert out["evidence"] == "[1] 售后退款: 7天无理由退货"
    assert out["citations"] == [
        {
            "n": 1,
            "chunk_id": 1,
            "section_path": "售后退款",
            "answer_snippet": "7天无理由退货",
        }
    ]


def test_create_ticket(sqlite_session_local):
    token = current_conversation_id.set(1)
    try:
        out = _invoke("create_ticket", {"description": "商品破损", "ticket_type": "售后"})
    finally:
        current_conversation_id.reset(token)

    assert isinstance(out, str)
    assert re.fullmatch(r"T\d{17}", out)

    with sqlite_session_local() as s:
        t = s.get(Ticket, out)
        assert t is not None
        assert t.conversation_id == 1
        assert t.description == "商品破损"
        assert t.ticket_type == "售后"
        assert t.status == "待处理"
