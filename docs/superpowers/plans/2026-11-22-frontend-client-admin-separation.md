# 前端客户端 / 管理端隔离 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把聊天页(顾客端)与管理后台(管理端)在前端彻底隔离：聊天页去掉后台导航，管理端收进统一侧栏外壳，后端零改动。

**Architecture:** 纯静态前端，无框架、无构建。新建 `static/admin-shell.js` 作为管理端导航的单一来源，读取 `<body data-page>` 注入统一侧栏；8 个管理页各自删掉手抄 `topnav`、加 `data-page` 并引入该脚本；聊天页 `index.html` 删掉 `topnav`、在侧栏底部加「管理后台 →」链接；`style.css` 加 `.admin-side` 侧栏样式并删除废弃的 `.topnav` 规则。

**Tech Stack:** 原生 HTML + JS + CSS（FastAPI `StaticFiles` 托管）；测试用 FastAPI `TestClient`（现有 `tests/test_static.py` 同款）。

**Spec:** `docs/superpowers/specs/2026-11-22-frontend-client-admin-separation-design.md`

## Global Constraints

- 后端不动：`app/main.py` 的路由与 `StaticFiles` mount 原样，不新增/删除任何后端路由。
- 不做登录/鉴权（内部演示）。
- 管理端导航固定为五项：后台总览 `/admin`、知识库 `/kb`、待审队列 `/review`、主题分布 `/topics`、验收 `/acceptance`，外加顶部「← 返回聊天」与「管理端」标识。
- `data-page` 取值 = 导航项 `href` 去掉前导 `/`：`admin` / `kb` / `review` / `topics` / `acceptance`；验收四个子页统一 `data-page="acceptance"`。
- 沿用现有配色变量（翡翠绿 `--brand: #059669` + 活力橙 `--accent: #ea580c`），管理端侧栏偏沉稳深色系。

## Review Focus

1. **顾客在聊天页看到后台入口** —— 聊天页不得出现任何后台导航链接，只能有侧栏底部一个不显眼的「管理后台」。测试：`test_chat_page_isolated`（Task 3）断言 `/` 无 `topnav`、无 `href="/kb"`、含「管理后台」。
2. **管理端回不去聊天** —— 侧栏顶部必须保留「← 返回聊天」。测试：`test_admin_shell_contains_nav`（Task 1）断言 `admin-shell.js` 文本含「返回聊天」。
3. **导航再次漂移（某页漏项）** —— 导航收敛到单一 JS 源，所有管理页必须引用它而非各自手抄。测试：`test_admin_pages_use_shell`（Task 2）断言 8 个管理页都引用 `admin-shell.js`、都有 `data-page`、都无 `topnav`；`test_admin_shell_contains_nav` 断言 JS 含全部 5 项与 5 个 href。
4. **验收四页激活态错乱** —— 验收四个子页应统一高亮「验收」。测试：`test_acceptance_subpages_map_to_acceptance`（Task 2）断言三子页 `data-page="acceptance"`；激活映射逻辑由 `admin-shell.js` 用 `href === "/" + data-page` 判定（Task 1 实现）。
5. **侧栏无样式或旧 topnav 残留** —— 新侧栏要有样式，废弃 `.topnav` 要清掉。测试：`test_admin_side_css_present`（Task 4）断言 `style.css` 含 `.admin-side` 且不含 `.topnav`。

---

### Task 1: 管理端侧栏外壳 `admin-shell.js`

**Files:**
- Create: `static/admin-shell.js`
- Test: `tests/test_static.py`（追加两个测试）

**Interfaces:**
- Consumes: 无（首个任务）。
- Produces: `static/admin-shell.js` —— 在 `DOMContentLoaded` 时读取 `document.body.dataset.page`，构建 `<aside class="admin-side">` 并插入 `document.body` 首个子节点；侧栏含「管理端」标识、「← 返回聊天」链接（`href="/"`）与五项导航（后台总览/知识库/待审队列/主题分布/验收，`href` 依次 `/admin` `/kb` `/review` `/topics` `/acceptance`）；命中项（`item.href === "/" + data-page`）加 `class="active"`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_static.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_static.py::test_admin_shell_served tests/test_static.py::test_admin_shell_contains_nav -v`
Expected: 两测试 FAIL（`/admin-shell.js` 返回 404）。

- [ ] **Step 3: 实现 `static/admin-shell.js`**

按 Interfaces 描述编写：定义导航项数组、`DOMContentLoaded` 时读 `data-page`、拼 `<aside class="admin-side">`、`document.body.prepend(...)` 注入；激活态用 `href === "/" + data-page` 判定。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_static.py::test_admin_shell_served tests/test_static.py::test_admin_shell_contains_nav -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add static/admin-shell.js tests/test_static.py
git commit -m "feat(ui): 管理端侧栏外壳 admin-shell.js(单一导航源)"
```

---

### Task 2: 管理页换共享外壳

**Files:**
- Modify: `static/admin.html`、`static/kb.html`、`static/review.html`、`static/topics.html`、`static/acceptance.html`、`static/acceptance_eval.html`、`static/acceptance_data.html`、`static/acceptance_errors.html`（各删 `<nav class="topnav">…</nav>`，给 `<body>` 加 `data-page`，在 `<head>` 或末尾加 `<script src="/admin-shell.js" defer></script>`）
- Test: `tests/test_static.py`（替换 `test_pages_have_topnav` 为新的两个测试）

**Interfaces:**
- Consumes: `static/admin-shell.js`（Task 1）。
- Produces: 各管理页 `<body data-page="…">` 取值——`admin.html`=`admin`、`kb.html`=`kb`、`review.html`=`review`、`topics.html`=`topics`、`acceptance.html` 及三个验收子页=`acceptance`；每页含 `<script src="/admin-shell.js" defer></script>` 且不含 `<nav class="topnav">`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_static.py` 中删除 `test_pages_have_topnav`，替换为：

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_static.py::test_admin_pages_use_shell tests/test_static.py::test_acceptance_subpages_map_to_acceptance -v`
Expected: FAIL（各管理页仍含 `topnav`、无 `data-page`）。

- [ ] **Step 3: 迁移 8 个管理页**

对每个文件：删除整段 `<nav class="topnav">…</nav>`；`<body class="kb-body">` 改为 `<body class="kb-body" data-page="<对应值>">`；加 `<script src="/admin-shell.js" defer></script>`（与页面自有 `*.js` 并列）。注意 `acceptance*.html` 是单行压缩写法，改时保持结构完整。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_static.py -v`
Expected: 全 PASS（含原有 `test_kb_page_served_with_blocks`、`test_review_page_served` 等仍通过）。

- [ ] **Step 5: 提交**

```bash
git add static/admin.html static/kb.html static/review.html static/topics.html static/acceptance.html static/acceptance_eval.html static/acceptance_data.html static/acceptance_errors.html tests/test_static.py
git commit -m "feat(ui): 8 个管理页换共享 admin-shell,移除手抄 topnav"
```

---

### Task 3: 聊天页客户端隔离

**Files:**
- Modify: `static/index.html`（删 `<nav class="topnav">…</nav>`，在侧栏 `<aside class="sidebar">` 底部、`#conversations` 之后加「管理后台 →」链接）
- Test: `tests/test_static.py`（追加 `test_chat_page_isolated`）

**Interfaces:**
- Consumes: 无（`index.html` 不引用 `admin-shell.js`）。
- Produces: `/` 页面不含 `topnav`、不含 `href="/kb"`，含「管理后台」文本与指向 `/admin` 的链接。

- [ ] **Step 1: 写失败测试**

在 `tests/test_static.py` 追加：

```python
def test_chat_page_isolated():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    assert 'class="topnav"' not in r.text
    assert 'href="/kb"' not in r.text
    assert "管理后台" in r.text
    assert 'href="/admin"' in r.text
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_static.py::test_chat_page_isolated -v`
Expected: FAIL（`/` 仍含 `topnav` 与 `href="/kb"`）。

- [ ] **Step 3: 改 `static/index.html`**

删除 `<nav class="topnav">…</nav>`；在 `#conversations` 之后、`</aside>` 之前加一个指向 `/admin` 的「管理后台 →」链接（用现有 `.new-chat` 或新加一个类，样式放 Task 4）。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_static.py::test_chat_page_isolated -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add static/index.html tests/test_static.py
git commit -m "feat(ui): 聊天页隔离,去掉后台导航仅留管理后台入口"
```

---

### Task 4: 管理端侧栏样式 + 清理废弃 topnav

**Files:**
- Modify: `static/style.css`（新增 `.admin-side` / `.admin-side a` / `.admin-side a.active` / `.admin-badge` 等；删除 `.topnav` 及其 `a`/`:hover`/`.active` 四条规则；给管理页 flex 布局补 `gap`）
- Test: `tests/test_static.py`（追加 `test_admin_side_css_present`）

**Interfaces:**
- Consumes: `admin-shell.js` 注入的 `<aside class="admin-side">` 结构与 `data-page` 激活 `class="active"`（Task 1）。
- Produces: `style.css` 含 `.admin-side`、不含 `.topnav`；侧栏视觉与管理端「管理端」标识样式落地。

- [ ] **Step 1: 写失败测试**

在 `tests/test_static.py` 追加：

```python
def test_admin_side_css_present():
    with TestClient(app) as c:
        r = c.get("/style.css")
    assert r.status_code == 200
    assert ".admin-side" in r.text
    assert ".topnav" not in r.text
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_static.py::test_admin_side_css_present -v`
Expected: FAIL（`style.css` 无 `.admin-side`、仍有 `.topnav`）。

- [ ] **Step 3: 改 `static/style.css`**

新增 `.admin-side`（镜像 `.sidebar` 的卡片观感，宽度约 200px、深色/沉稳配色以区别于聊天页），`.admin-side a` 链接样式、`.admin-side a.active` 高亮（复用 `--brand` 或更深色）、`.admin-badge`「管理端」标识样式；删除 `.topnav` 四条规则；在 `.kb-body` 加 `gap: 16px` 让侧栏与 `.kb-card` 有间距。

- [ ] **Step 4: 运行确认通过 + 全量回归**

Run: `pytest tests/test_static.py -v` 然后 `pytest -q`
Expected: `test_static.py` 全 PASS，全量 `pytest -q` 绿（现有 ~329 项 + 新增项，无回归）。

- [ ] **Step 5: 提交**

```bash
git add static/style.css tests/test_static.py
git commit -m "style(ui): 管理端侧栏样式,清理废弃 topnav"
```
