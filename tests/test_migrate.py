"""Tests for the on-demand MySQL schema migration CLI (no real engine touched)."""

from sqlalchemy import create_engine

from app.jobs.migrate import ensure_tables


def test_ensure_tables_creates_missing_and_reports_them():
    engine = create_engine("sqlite:///:memory:")

    ensured = ensure_tables(engine)

    assert "low_confidence_questions" in ensured
    assert "faith_cases" in ensured
    assert "knowledge_chunks" in ensured

    # Idempotent: a second run has nothing left to create.
    assert ensure_tables(engine) == []
