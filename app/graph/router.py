"""Intent → route mapping (hard-coded, ch06)."""

from __future__ import annotations

# 八类意图(七类 + 「其他」) → 五出口(权威方案 §3.1):
#   escalate        = 投诉(安抚话术 + 转人工/建工单建议)
#   fallback_script = 闲聊、其他(script_reply 按意图分文案,零模型调用)
#   knowledge       = 商品咨询(检索 + 置信度闸)
#   refund_flow     = 退款退货、售后(确定性子流程:取单 interrupt→检索政策→Agent)
#   business        = 物流、订单(直接进主力 Agent 调工具)
_ROUTE_BY_INTENT = {
    "投诉": "escalate",
    "闲聊": "fallback_script",
    "其他": "fallback_script",
    "商品咨询": "knowledge",
    "退款退货": "refund_flow",
    "售后": "refund_flow",
    "物流": "business",
    "订单": "business",
    "建工单": "business",
}


def route_from_intent(intent: str) -> str:
    """Return the route for an intent, raising ``ValueError`` on unknown."""
    if intent not in _ROUTE_BY_INTENT:
        raise ValueError(f"unknown intent: {intent}")
    return _ROUTE_BY_INTENT[intent]
