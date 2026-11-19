"""Intent → route mapping (hard-coded, ch05)."""

from __future__ import annotations

# 七类意图 → 四出口:
#   knowledge  = 商品咨询、退款退货(强制 RAG 检索 + 置信度闸)
#   business   = 物流、订单、售后(直接进主力 Agent 调工具)
#   complaint  = 投诉(安抚话术 + 转人工/建工单建议)
#   chitchat   = 闲聊(固定话术,不花模型调用)
_ROUTE_BY_INTENT = {
    "商品咨询": "knowledge",
    "退款退货": "knowledge",
    "物流": "business",
    "订单": "business",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "chitchat",
}


def route_from_intent(intent: str) -> str:
    """Return the route for an intent, raising ``ValueError`` on unknown."""
    if intent not in _ROUTE_BY_INTENT:
        raise ValueError(f"unknown intent: {intent}")
    return _ROUTE_BY_INTENT[intent]
