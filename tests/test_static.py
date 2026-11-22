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


def test_admin_pages_use_shell():
    paths = ("/kb", "/admin", "/review", "/topics", "/acceptance",
             "/acceptance/eval", "/acceptance/data", "/acceptance/errors")
    for path in paths:
        with TestClient(app) as c:
            r = c.get(path)
        assert r.status_code == 200
        assert "admin-shell.js" in r.text
        assert "data-page=" in r.text
        assert 'class="topnav"' not in r.text


def test_acceptance_subpages_map_to_acceptance():
    for path in ("/acceptance/eval", "/acceptance/data", "/acceptance/errors"):
        with TestClient(app) as c:
            r = c.get(path)
        assert 'data-page="acceptance"' in r.text


def test_admin_shell_served():
    with TestClient(app) as c:
        r = c.get("/admin-shell.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_admin_shell_contains_nav():
    with TestClient(app) as c:
        r = c.get("/admin-shell.js")
    text = r.text
    for label in ("返回聊天", "管理端", "后台总览", "知识库", "待审队列", "主题分布", "验收"):
        assert label in text
    for href in ("/admin", "/kb", "/review", "/topics", "/acceptance"):
        assert f'href="{href}"' in text


def test_chat_page_isolated():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    assert 'class="topnav"' not in r.text
    assert 'href="/kb"' not in r.text
    assert "管理后台" in r.text
    assert 'href="/admin"' in r.text


def test_admin_side_css_present():
    with TestClient(app) as c:
        r = c.get("/style.css")
    assert r.status_code == 200
    assert ".admin-side" in r.text
    assert ".topnav" not in r.text


def test_admin_side_responsive():
    with TestClient(app) as c:
        r = c.get("/style.css")
    assert r.status_code == 200
    # 窄屏下管理端侧栏横排(flex-direction: row),避免固定 200px 挤压内容
    assert "flex-direction: row" in r.text
