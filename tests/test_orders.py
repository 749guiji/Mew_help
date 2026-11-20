"""Tests for the ch06 mock order data source + /api/orders."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_seed_orders_shape():
    from app.orders import list_orders

    orders = list_orders()
    assert len(orders) >= 5
    for o in orders:
        assert {"order_id", "product", "amount", "status"} <= o.keys()


def test_get_order_hit_and_miss():
    from app.orders import get_order

    assert get_order("1001")["product"]
    assert get_order("nope") is None


class _FakeExtraction:
    async def extract(self, text):
        return None


def test_orders_endpoint():
    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.get("/api/orders")
    assert r.status_code == 200
    assert len(r.json()["orders"]) >= 5
