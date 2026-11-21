"""Tests for ch09 Langfuse observability wiring (unconfigured = silent skip)."""

import app.observability as obs


def test_handler_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(obs, "langfuse_enabled", lambda: False)
    assert obs.get_langfuse_handler() is None


def test_handler_returned_when_configured(monkeypatch):
    monkeypatch.setattr(obs, "langfuse_enabled", lambda: True)
    assert obs.get_langfuse_handler() is not None


def test_trace_url_none_for_empty_id():
    assert obs.langfuse_trace_url(None) == ""
