"""Tests for the deterministic confidence gate (app.graph.nodes)."""

from app.graph.nodes import gate_pass


def test_gate_pass_above_threshold():
    assert gate_pass(0.6, 0.5) is True


def test_gate_fail_below():
    assert gate_pass(0.4, 0.5) is False


def test_gate_fail_on_none():
    assert gate_pass(None, 0.5) is False
