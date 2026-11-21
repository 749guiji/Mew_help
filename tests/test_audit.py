"""Tests for the tool audit logger (ch08)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.tools.audit as audit
from app.db import Base
from app.db.models import ToolAuditLog
from app.tools.audit import log_tool_audit


@pytest.fixture
def audit_session_local(monkeypatch):
    """Swap app.tools.audit.SessionLocal for a SQLite in-memory sessionmaker."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(audit, "SessionLocal", SessionLocal)
    return SessionLocal


def test_log_audit_writes_row(audit_session_local):
    log_tool_audit(
        tool_name="query_order",
        tool_source="builtin",
        status="成功",
        retry_count=1,
        duration_ms=12,
    )
    with audit_session_local() as s:
        row = s.query(ToolAuditLog).one()
        assert row.status == "成功"
        assert row.retry_count == 1
        assert row.tool_source == "builtin"
        assert row.duration_ms == 12


def test_log_audit_failure_does_not_raise(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(audit, "SessionLocal", boom)
    # 审计写入失败不能抛,直接返回
    log_tool_audit(tool_name="x", tool_source="builtin", status="成功")
