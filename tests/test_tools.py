"""Tests for the chapter 3 tool functions and registry."""

import re

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.tools.db_tools as db_tools
from app.db import Base
from app.db.models import Ticket
from app.db.seed import seed_faq
from app.tools import ALL_TOOLS, TOOL_REGISTRY
from app.tools.context import current_conversation_id


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


def test_registry_has_five_tools():
    assert set(TOOL_REGISTRY) == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
    assert len(ALL_TOOLS) == 5


def test_query_order_returns_str():
    out = TOOL_REGISTRY["query_order"].invoke({"order_id": "1001"})
    assert isinstance(out, str) and "1001" in out


def test_query_product_returns_str():
    out = TOOL_REGISTRY["query_product"].invoke({"product_name": "保温杯"})
    assert isinstance(out, str) and "保温杯" in out


def test_query_logistics_returns_str():
    out = TOOL_REGISTRY["query_logistics"].invoke({"order_id": "1002"})
    assert isinstance(out, str) and "1002" in out


def test_query_faq_no_match(sqlite_session_local):
    out = TOOL_REGISTRY["query_faq"].invoke({"keyword": "邮费"})
    assert "未查到" in out


def test_query_faq_match(sqlite_session_local):
    out = TOOL_REGISTRY["query_faq"].invoke({"keyword": "退货"})
    assert "退货" in out


def test_create_ticket(sqlite_session_local):
    token = current_conversation_id.set(1)
    try:
        out = TOOL_REGISTRY["create_ticket"].invoke(
            {"description": "商品破损", "ticket_type": "售后"}
        )
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
