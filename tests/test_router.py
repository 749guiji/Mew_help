"""Tests for the intent→route mapping (app.graph.router)."""

import pytest

from app.graph.router import route_from_intent


def test_knowledge_intents():
    assert route_from_intent("商品咨询") == "knowledge"
    assert route_from_intent("退款退货") == "knowledge"


def test_business_intents():
    assert route_from_intent("物流") == "business"
    assert route_from_intent("订单") == "business"
    assert route_from_intent("售后") == "business"


def test_complaint_chitchat():
    assert route_from_intent("投诉") == "complaint"
    assert route_from_intent("闲聊") == "chitchat"


def test_unknown_intent_raises():
    with pytest.raises(ValueError):
        route_from_intent("未知意图")
