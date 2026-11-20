"""Mock order data source (ch06).

Orders are an in-process constant list (matching ch02's "订单走工具内 mock,
不建表" decision), not a database table. The order picker and ``query_order``
tool both read from here, so the list the user selects from and the list the
tool answers from never diverge.
"""

from __future__ import annotations

SEED_ORDERS: list[dict] = [
    {"order_id": "1001", "product": "无线蓝牙耳机", "amount": "199.00", "status": "已签收"},
    {"order_id": "1002", "product": "保温杯", "amount": "89.00", "status": "已发货，运输中"},
    {"order_id": "1003", "product": "机械键盘", "amount": "399.00", "status": "已完成"},
    {"order_id": "1004", "product": "帆布鞋", "amount": "159.00", "status": "已支付，待发货"},
    {"order_id": "1005", "product": "充电宝", "amount": "129.00", "status": "退款中"},
]


def list_orders() -> list[dict]:
    """Return a copy of the seed orders (callers must not mutate the source)."""
    return [dict(o) for o in SEED_ORDERS]


def get_order(order_id: str) -> dict | None:
    """Return the order with ``order_id``, or ``None`` when absent."""
    for o in SEED_ORDERS:
        if o["order_id"] == order_id:
            return dict(o)
    return None
