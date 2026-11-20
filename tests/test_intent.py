"""Tests for intent classification + coreference passthrough (app.graph.nodes)."""

from types import SimpleNamespace

from app.graph.nodes import IntentResult, classify_intent, maybe_escalate, resolve_coref


class FakeLLM:
    """Callable stand-in coerced to a RunnableLambda by ``|``."""

    def __init__(self, content):
        self._content = content

    def __call__(self, inp):
        return SimpleNamespace(content=self._content)


def test_classify_intent_parses_json():
    r = classify_intent(FakeLLM('{"intent":"退款退货","confidence":0.9}'), "这个能退吗")
    assert r.intent == "退款退货"
    assert r.confidence == 0.9


def test_classify_intent_falls_back_to_other_on_bad_json():
    r = classify_intent(FakeLLM("我不知道"), "这个能退吗")
    assert r.intent == "其他"
    assert r.confidence == 0.0


def test_classify_intent_falls_back_on_missing_field():
    r = classify_intent(FakeLLM('{"intent":"退款退货"}'), "这个能退吗")
    assert r.intent == "其他"
    assert r.confidence == 0.0


def test_maybe_escalate_noop_without_escalate_llm():
    r = IntentResult("物流", 0.4)
    assert maybe_escalate(None, "它到哪了", r, 0.6) is r


def test_maybe_escalate_keeps_high_confidence():
    r = IntentResult("物流", 0.9)
    out = maybe_escalate(FakeLLM('{"intent":"物流","confidence":0.9}'), "它到哪了", r, 0.6)
    assert out is r


def test_maybe_escalate_escalates_low_confidence():
    r = IntentResult("物流", 0.4)
    big = FakeLLM('{"intent":"退款退货","confidence":0.85}')
    out = maybe_escalate(big, "它能退吗", r, 0.6)
    assert out.intent == "退款退货"
    assert out.confidence == 0.85


def test_resolve_coref_passthrough_when_complete():
    assert resolve_coref(FakeLLM("物流到哪了"), "物流到哪了", []) == "物流到哪了"


def test_resolve_coref_falls_back_on_empty():
    assert resolve_coref(FakeLLM(""), "它能退吗", ["我买了耳机"]) == "它能退吗"


def test_resolve_coref_returns_model_output():
    assert resolve_coref(FakeLLM("耳机能退货吗"), "它能退吗", ["我买了耳机"]) == "耳机能退货吗"
