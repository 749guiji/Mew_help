"""Tests for the /admin aggregate homepage."""

from fastapi.testclient import TestClient

import app.api.routes.admin as admin
from app.main import create_app


def test_admin_page_served():
    with TestClient(create_app()) as c:
        r = c.get("/admin")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "后台" in r.text


def test_admin_overview_returns_kb_card(monkeypatch):
    monkeypatch.setattr(
        admin,
        "CARD_BUILDERS",
        [
            lambda: {
                "key": "kb",
                "title": "知识库",
                "metrics": {"chunks": 3, "pending": 1},
                "error": None,
            }
        ],
    )
    with TestClient(create_app()) as c:
        r = c.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert len(cards) == 1
    assert cards[0]["key"] == "kb"
    assert cards[0]["metrics"]["chunks"] == 3
