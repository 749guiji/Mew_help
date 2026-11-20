"""Tests for the ch06 mock order data source (per-user stable)."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_list_user_orders_shape():
    from app.orders import list_user_orders

    orders = list_user_orders("u1")
    assert 2 <= len(orders) <= 4
    for o in orders:
        assert {"order_id", "product", "amount", "status"} <= o.keys()


def test_list_user_orders_stable():
    from app.orders import list_user_orders

    assert list_user_orders("u1") == list_user_orders("u1")


def test_order_snapshot_stable():
    from app.orders import order_snapshot

    a = order_snapshot("1001")
    b = order_snapshot("1001")
    assert a == b
    assert a["order_id"] == "1001"
    assert {"product", "status", "amount", "tracking_no"} <= a.keys()


class _FakeExtraction:
    async def extract(self, text):
        return None


def test_orders_endpoint():
    app = create_app(extraction_service=_FakeExtraction())
    with TestClient(app) as c:
        r = c.get("/api/orders", params={"user_id": "u1"})
    assert r.status_code == 200
    assert 2 <= len(r.json()["orders"]) <= 4
