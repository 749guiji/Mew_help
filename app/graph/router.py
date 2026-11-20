"""Intent → route mapping (hard-coded, ch06)."""

from __future__ import annotations

# 七类意图 + 「其他」 → 六出口:
#   knowledge  = 商品咨询(单查询检索 + 置信度闸)
#   refund     = 退款退货、售后(确定性子流程:拿订单→扩写→检索政策→Agent)
#   business   = 物流、订单、其他(直接进主力 Agent 调工具)
#   complaint  = 投诉(安抚话术 + 转人工/建工单建议)
#   chitchat   = 闲聊(固定话术,不花模型调用)
_ROUTE_BY_INTENT = {
    "商品咨询": "knowledge",
    "退款退货": "refund",
    "物流": "business",
    "订单": "business",
    "售后": "refund",
    "投诉": "complaint",
    "闲聊": "chitchat",
    "其他": "business",
}


def route_from_intent(intent: str) -> str:
    """Return the route for an intent, raising ``ValueError`` on unknown."""
    if intent not in _ROUTE_BY_INTENT:
        raise ValueError(f"unknown intent: {intent}")
    return _ROUTE_BY_INTENT[intent]
