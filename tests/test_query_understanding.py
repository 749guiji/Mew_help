"""Tests for ch04 query understanding: LLM rewrite + retrieval-time synonyms."""

from app.kb.chunker import chunk_markdown
from app.kb.query_understanding import (
    QueryUnderstanding,
    SYNONYM_DICT,
    expand_synonyms,
    rewrite_query,
)


class FakeStructuredRunnable:
    """Callable stand-in coerced to a ``RunnableLambda`` by ``|``."""

    def __init__(self, result):
        self._result = result

    def __call__(self, inp):
        return self._result


class FakeLLM:
    """Records ``method`` and returns a runnable yielding a canned result."""

    def __init__(self, result):
        self._result = result
        self.called_model = None
        self.called_method = None

    def with_structured_output(self, model, method=None):
        self.called_model = model
        self.called_method = method
        return FakeStructuredRunnable(self._result)


def test_rewrite_query_returns_model_and_uses_function_calling():
    canned = QueryUnderstanding(standard_question="手机壳如何退货", category="退货政策")
    llm = FakeLLM(canned)

    out = rewrite_query(llm, "手机壳咋退货啊")

    assert isinstance(out, QueryUnderstanding)
    assert out.standard_question == "手机壳如何退货"
    assert out.category == "退货政策"
    assert llm.called_model is QueryUnderstanding
    assert llm.called_method == "function_calling"


def test_rewrite_query_falls_back_when_chain_returns_none():
    llm = FakeLLM(None)

    out = rewrite_query(llm, "运费怎么算")

    assert isinstance(out, QueryUnderstanding)
    assert out.standard_question == "运费怎么算"
    assert out.category is None


def test_expand_synonyms_keeps_original_and_appends():
    out = expand_synonyms("退款")
    assert "退款" in out
    assert "退钱" in out


def test_synonym_dict_is_pure_lookup_table():
    assert isinstance(SYNONYM_DICT, dict)
    assert "退款" in SYNONYM_DICT
    assert "退钱" in SYNONYM_DICT["退款"]


def test_expand_synonyms_does_not_create_extra_chunks():
    text = "# 退款政策\n\nQ: 怎么退款\nA: 联系客服申请退款"
    original = chunk_markdown(text, "faq")
    expanded = chunk_markdown(expand_synonyms(text), "faq")

    assert len(original) > 0
    assert len(expanded) == len(original)
