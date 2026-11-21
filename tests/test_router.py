"""Tests for the intent→route mapping (app.graph.router, 5 outlets)."""

import pytest

from app.graph.router import route_from_intent


def test_knowledge_intents():
    assert route_from_intent("商品咨询") == "knowledge"


def test_refund_flow_intents():
    assert route_from_intent("退款退货") == "refund_flow"
    assert route_from_intent("售后") == "refund_flow"


def test_business_intents():
    assert route_from_intent("物流") == "business"
    assert route_from_intent("订单") == "business"
    assert route_from_intent("建工单") == "business"


def test_escalate_intent():
    assert route_from_intent("投诉") == "escalate"


def test_fallback_script_intents():
    assert route_from_intent("闲聊") == "fallback_script"
    assert route_from_intent("其他") == "fallback_script"


def test_unknown_intent_raises():
    with pytest.raises(ValueError):
        route_from_intent("未知意图")
