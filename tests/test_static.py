from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_index():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "客服" in r.text


def test_style_served():
    with TestClient(app) as c:
        r = c.get("/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]


def test_script_served():
    with TestClient(app) as c:
        r = c.get("/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_chat_script_has_handoff_buttons():
    with TestClient(app) as c:
        r = c.get("/app.js")
    assert r.status_code == 200
    assert "转人工" in r.text
    assert "建工单" in r.text


def test_kb_page_served_with_blocks():
    with TestClient(app) as c:
        r = c.get("/kb")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    for heading in ("库存闸条", "对话挖知识", "最近入库", "检索自测", "手工录入"):
        assert heading in r.text


def test_kb_script_served():
    with TestClient(app) as c:
        r = c.get("/kb.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_admin_script_served():
    with TestClient(app) as c:
        r = c.get("/admin.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_review_page_served():
    with TestClient(app) as c:
        r = c.get("/review")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "待审队列" in r.text


def test_review_script_served():
    with TestClient(app) as c:
        r = c.get("/review.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_pages_have_topnav():
    for path in ("/", "/kb", "/admin", "/review"):
        with TestClient(app) as c:
            r = c.get(path)
        assert r.status_code == 200
        assert 'class="topnav"' in r.text
        for href in ("/kb", "/review", "/admin"):
            assert f'href="{href}"' in r.text
