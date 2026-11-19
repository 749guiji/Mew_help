"""Tests for intent classification + coreference passthrough (app.graph.nodes)."""

from types import SimpleNamespace

from app.graph.nodes import classify_intent, resolve_coref


class FakeLLM:
    """Callable stand-in coerced to a RunnableLambda by ``|``."""

    def __init__(self, content):
        self._content = content

    def __call__(self, inp):
        return SimpleNamespace(content=self._content)


def test_classify_intent_returns_matched_intent():
    assert classify_intent(FakeLLM("订单"), "订单 1001 到哪了") == "订单"


def test_classify_intent_handles_parenthetical():
    assert classify_intent(FakeLLM("订单(订单状态)"), "订单 1001") == "订单"


def test_classify_intent_falls_back_on_no_match():
    assert classify_intent(FakeLLM("我不知道"), "你好") == "闲聊"


def test_resolve_coref_passthrough():
    assert resolve_coref("它到哪了") == "它到哪了"
