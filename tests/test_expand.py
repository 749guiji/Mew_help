"""Tests for query expansion (app.graph.nodes.expand_queries)."""

from types import SimpleNamespace

from app.graph.nodes import expand_queries


class FakeLLM:
    """Callable stand-in coerced to a RunnableLambda by ``|``."""

    def __init__(self, content):
        self._content = content

    def __call__(self, inp):
        return SimpleNamespace(content=self._content)


def test_expand_parses_queries():
    assert expand_queries(
        FakeLLM('{"queries":["退款政策","退货条件"]}'), "这个能退吗"
    ) == ["退款政策", "退货条件"]


def test_expand_falls_back_on_bad_json():
    assert expand_queries(FakeLLM("不知道"), "这个能退吗") == ["这个能退吗"]


def test_expand_falls_back_on_empty_queries():
    assert expand_queries(FakeLLM('{"queries":[]}'), "这个能退吗") == ["这个能退吗"]


def test_expand_falls_back_on_missing_field():
    assert expand_queries(FakeLLM('{"foo":"bar"}'), "这个能退吗") == ["这个能退吗"]
