"""订单列表 API:给订单选择器与前端提供可点选的 mock 订单。"""

from __future__ import annotations

from fastapi import APIRouter

from app.orders import list_orders

router = APIRouter()


@router.get("/api/orders")
def get_orders() -> dict:
    """Return the seed order list as ``{"orders": [...]}``."""
    return {"orders": list_orders()}
