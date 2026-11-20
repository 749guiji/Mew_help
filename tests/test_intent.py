"""Tests for intent classification + coreference passthrough (app.graph.nodes)."""

from types import SimpleNamespace

from app.graph.nodes import IntentResult, classify_intent, resolve_coref


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


def test_classify_intent_unknown_intent_falls_back_to_other():
    r = classify_intent(FakeLLM('{"intent":"查快递","confidence":0.9}'), "这个能退吗")
    assert r.intent == "其他"
    assert r.confidence == 0.0


class CapturingLLM:
    def __init__(self, content):
        self._content = content
        self.last_input = None

    def __call__(self, inp):
        self.last_input = inp
        return SimpleNamespace(content=self._content)


def test_classify_intent_passes_history_to_prompt():
    llm = CapturingLLM('{"intent":"物流","confidence":0.9}')
    classify_intent(llm, "它到哪了", "用户:我买了耳机")
    assert "我买了耳机" in str(llm.last_input)


def test_resolve_coref_passthrough_when_complete():
    assert resolve_coref(FakeLLM("物流到哪了"), "物流到哪了", []) == "物流到哪了"


def test_resolve_coref_falls_back_on_empty():
    assert resolve_coref(FakeLLM(""), "它能退吗", ["我买了耳机"]) == "它能退吗"


def test_resolve_coref_returns_model_output():
    assert resolve_coref(FakeLLM("耳机能退货吗"), "它能退吗", ["我买了耳机"]) == "耳机能退货吗"
