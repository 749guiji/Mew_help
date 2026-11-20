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


def test_resolve_coref_passthrough_when_complete():
    assert resolve_coref(FakeLLM("物流到哪了"), "物流到哪了", []) == "物流到哪了"


def test_resolve_coref_falls_back_on_empty():
    assert resolve_coref(FakeLLM(""), "它能退吗", ["我买了耳机"]) == "它能退吗"


def test_resolve_coref_returns_model_output():
    assert resolve_coref(FakeLLM("耳机能退货吗"), "它能退吗", ["我买了耳机"]) == "耳机能退货吗"
