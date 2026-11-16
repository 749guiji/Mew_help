from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_index():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "客服" in r.text
