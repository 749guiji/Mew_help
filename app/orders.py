"""Mock order data source (ch06 rework): per-user stable orders.

订单不落库,用随机种子生成确定性的 mock 快照:同 order_id 永远同一单、
同 user_id 永远同一组订单。query_order 工具与 fetch_order 节点同源,
所以「订单选择器列的订单」和「点选后查到的订单」永远对得上。
"""

from __future__ import annotations

import random

_PRODUCTS = ["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]
_STATUSES = ["待付款", "已付款", "已发货", "已签收"]


def order_snapshot(order_id: str) -> dict:
    """订单快照(纯函数,种子固定 → 同 order_id 稳定)。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "product": rng.choice(_PRODUCTS),
        "status": rng.choice(_STATUSES),
        "amount": rng.randint(50, 2000),
        "tracking_no": f"SF{rng.randint(10**11, 10**12 - 1)}",
    }


def list_user_orders(user_id: str) -> list[dict]:
    """按 user_id 稳定列出该用户的 2~4 笔订单(mock,不落库)。"""
    rng = random.Random(f"user_orders:{user_id}")
    ids = [str(rng.randint(1000, 9999)) for _ in range(rng.randint(2, 4))]
    out: list[dict] = []
    for oid in ids:
        s = order_snapshot(oid)
        out.append(
            {"order_id": oid, "product": s["product"], "status": s["status"], "amount": s["amount"]}
        )
    return out


def get_order(order_id: str) -> dict:
    """返回 ``order_snapshot(order_id)``(mock 下任何订单号都命中)。"""
    return order_snapshot(order_id)
