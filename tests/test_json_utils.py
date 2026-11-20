"""Tests for the robust JSON object parser (app.graph.json_utils)."""

from app.graph.json_utils import parse_json_object


def test_parse_clean_json():
    assert parse_json_object('{"intent":"退款退货","confidence":0.9}') == {
        "intent": "退款退货",
        "confidence": 0.9,
    }


def test_parse_with_markdown_fence():
    assert parse_json_object('```json\n{"queries":["a","b"]}\n```') == {
        "queries": ["a", "b"]
    }


def test_parse_garbage_returns_none():
    assert parse_json_object("我不知道") is None
