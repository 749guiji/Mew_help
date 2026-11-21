"""Interrupt/resume tests for the ch08 ticket-confirmation flow (confirm_ticket)."""

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.graph import agent, nodes
from app.graph.build import build_graph
from app.graph.nodes import IntentResult


def _build(monkeypatch, pending):
    monkeypatch.setattr(nodes, "resolve_coref", lambda llm, q, h: q)
    monkeypatch.setattr(
        nodes, "classify_intent", lambda llm, q, h="": IntentResult("建工单", 0.9)
    )

    async def fake_agent(state):
        return {
            "messages": [("assistant", "")],
            "pending_ticket": pending,
            "visited": ["agent"],
        }

    monkeypatch.setattr(agent, "react_agent_node", fake_agent)
    return build_graph()


def _invoke(graph, query, thread, **extra):
    config = {"configurable": {"thread_id": thread}}
    return asyncio.run(
        graph.ainvoke({"messages": [("user", query)], "raw_query": query, **extra}, config)
    )


def _resume(graph, payload, thread):
    config = {"configurable": {"thread_id": thread}}
    return asyncio.run(graph.ainvoke(Command(resume=payload), config))


@pytest.fixture
def sqlite(monkeypatch):
    """把 create_ticket 写库 / 审计写库 / 上下文元数据三处 SessionLocal 都换 SQLite。

    用 StaticPool 共享单连接:LangGraph 把 sync 节点跑在 worker 线程,而 SQLite
    内存库默认每线程独立,建表与查询不在同一线程会看不到表。
    """
    import app.graph.context_meta as context_meta
    import app.tools.audit as audit
    import app.tools.db_tools as db_tools
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.db import Base

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SL = sessionmaker(bind=engine)
    monkeypatch.setattr(db_tools, "SessionLocal", SL)
    monkeypatch.setattr(audit, "SessionLocal", SL)
    monkeypatch.setattr(context_meta, "SessionLocal", SL)
    return SL


_PENDING = {"tool_call_id": "c1", "args": {"description": "商品破损", "ticket_type": "售后"}}


def test_interrupt_pushes_ticket_preview(monkeypatch):
    graph = _build(monkeypatch, _PENDING)
    result = _invoke(graph, "帮我建个工单", "t-preview")

    intr = result.get("__interrupt__")
    assert intr and intr[0].value["type"] == "ticket_preview"
    assert intr[0].value["draft"]["ticket_type"] == "售后"
    assert intr[0].value["draft"]["description"] == "商品破损"


def test_resume_confirm_writes_ticket_and_audit(monkeypatch, sqlite):
    from app.db.models import Ticket, ToolAuditLog

    graph = _build(monkeypatch, _PENDING)
    r1 = _invoke(graph, "帮我建个工单", "t-confirm", conversation_id=1)
    assert r1.get("__interrupt__")

    r2 = _resume(graph, {"confirm": True}, "t-confirm")

    answers = [m for m in r2["messages"] if isinstance(m, AIMessage) and m.content]
    assert any("工单号" in m.content for m in answers)

    with sqlite() as s:
        tickets = s.query(Ticket).all()
        assert len(tickets) == 1
        assert tickets[0].ticket_type == "售后"
        assert tickets[0].conversation_id == 1
        audits = s.query(ToolAuditLog).all()
        assert any(a.status == "成功" and a.tool_name == "create_ticket" for a in audits)


def test_resume_cancel_no_ticket_and_denied_audit(monkeypatch, sqlite):
    from app.db.models import Ticket, ToolAuditLog

    graph = _build(monkeypatch, _PENDING)
    r1 = _invoke(graph, "帮我建个工单", "t-cancel", conversation_id=1)
    assert r1.get("__interrupt__")

    r2 = _resume(graph, {"confirm": False}, "t-cancel")

    answers = [m for m in r2["messages"] if isinstance(m, AIMessage) and m.content]
    assert any("取消" in m.content for m in answers)

    with sqlite() as s:
        assert s.query(Ticket).count() == 0  # 没建单
        audits = s.query(ToolAuditLog).all()
        assert any(a.status == "权限拒绝" and a.tool_name == "create_ticket" for a in audits)


def test_resume_wrong_shape_does_not_crash(monkeypatch, sqlite):
    from app.db.models import Ticket

    graph = _build(monkeypatch, _PENDING)
    r1 = _invoke(graph, "帮我建个工单", "t-wrong", conversation_id=1)
    assert r1.get("__interrupt__")

    r2 = _resume(graph, "意外的字符串", "t-wrong")  # 形状不对 → 当取消,不崩

    with sqlite() as s:
        assert s.query(Ticket).count() == 0
