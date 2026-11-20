"""订单列表 API:给订单选择器与前端提供可点选的 mock 订单。"""

from __future__ import annotations

from fastapi import APIRouter

from app.orders import list_user_orders

router = APIRouter()


@router.get("/api/orders")
def get_orders(user_id: str = "guest") -> dict:
    """按 user_id 返回稳定的 mock 订单列表 ``{"orders": [...]}``。"""
    return {"orders": list_user_orders(user_id)}
