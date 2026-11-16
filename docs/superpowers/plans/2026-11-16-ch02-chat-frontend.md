# 第二章「聊天前端」Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给已完成的纯对话后端加一个暖橙客服聊天网页,浏览器打开即聊,SSE 回复逐字渲染,支持多轮。

**Architecture:** 纯静态 `static/` 三件套(`index.html`/`style.css`/`app.js`),由 FastAPI 在 `create_app` 末尾 `mount("/", StaticFiles(html=True))` 同源托管。前端用 `fetch` + `ReadableStream` 解析 `/chat` 的 SSE(非 EventSource,因 `/chat` 是 POST)。

**Tech Stack:** FastAPI 0.163 + StaticFiles · 原生 HTML/CSS/JS(零 npm/构建/框架)· pytest(后端)。

**Spec:** `docs/superpowers/specs/2026-11-16-ch02-chat-frontend-design.md`

## Global Constraints

- 纯静态零依赖:不引入 npm / 构建工具 / 框架,不新增 Python 依赖。
- 不改 `/chat`、`/extract` 业务逻辑,只在 `create_app` 挂静态。
- 静态目录用绝对路径解析(不依赖 CWD):`STATIC_DIR = Path(__file__).resolve().parent.parent / "static"`。
- 暖橙配色色值:`--bg #faf5ef`、`--brand #f97316`、`--brand-hover #ea580c`、`--assistant-bubble #fff7ed`、`--text #2f2a26`、`--muted #8a8178`、`--online #22c55e`。
- 中文文案:标题「在线客服」、状态「在线」、欢迎语「您好,我是智能客服,请问有什么可以帮您?」。
- commit message 末尾附 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。

## Review Focus

1. **流式期间重复发送** → 前端需用 `streaming` 标志 + 禁用按钮拒绝并发,否则产生并发请求/错乱气泡。
2. **SSE 半帧(chunk 边界截断)** → 必须用持久 buffer 累积、只在 `\n\n` 处切帧,半帧留到下一块。
3. **空/纯空白消息** → 前端 trim 后拒发(后端 422 兜底已有 `test_chat_empty_message_422`)。
4. **多轮 session_id 未传递** → 首个 `meta` 事件的 `session_id` 必须存下,后续请求带上,否则追问上下文断。
5. **静态资源 404 / MIME 错** → 用 pytest 断言 `GET /style.css`→200+`text/css`、`GET /app.js`→200+`application/javascript`。

---

### Task 1: 后端静态托管(挂载 + 占位页)

**Files:**
- Modify: `app/main.py`
- Create: `static/index.html`(占位,Task 2 替换)
- Test: `tests/test_static.py`

**Interfaces:**
- Produces:`create_app()` 现挂载 `StaticFiles`;模块级 `STATIC_DIR = Path(__file__).resolve().parent.parent / "static"`;`/chat`、`/extract`、`/docs` 先注册、静态 mount 兜底,互不冲突。

- [ ] **Step 1: 写失败测试**

`tests/test_static.py`(用模块级 `app` 即可,GET / 不触服务):

```python
from fastapi.testclient import TestClient
from app.main import app

def test_root_serves_index():
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "客服" in r.text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_static.py -v`
Expected: FAIL(当前 `/` 返回 404)

- [ ] **Step 3: 实现挂载 + 占位页**

`app/main.py`:`from pathlib import Path`、`from fastapi.staticfiles import StaticFiles`;定义 `STATIC_DIR = Path(__file__).resolve().parent.parent / "static"`;在 `include_router(extract_routes.router)` 之后、`return app` 之前加 `app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")`。

`static/index.html` 占位:`<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>客服</title></head><body>客服聊天</body></html>`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_static.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py static/index.html tests/test_static.py
git commit -m "feat: serve static frontend from FastAPI"
```

---

### Task 2: 聊天页前端(HTML/CSS/JS + SSE 流式渲染)

**Files:**
- Modify: `static/index.html`(替换占位为完整页)
- Create: `static/style.css`
- Create: `static/app.js`
- Modify: `tests/test_static.py`(加静态资源断言)

**Interfaces:**
- Consumes:`/chat` SSE 事件序列:`{"type":"meta","session_id":...}` → 多个 `{"type":"token","content":...}` → `{"type":"done"}`,异常时 `{"type":"error","message":...}`。
- Produces:前端状态 `state = { sessionId: null, streaming: false }` + 模块级 `sseBuffer = ""`;`app.js` 暴露 `send()` 绑定到发送按钮/Enter。

- [ ] **Step 1: 写失败测试(静态资源)**

`tests/test_static.py` 追加:

```python
def test_style_served():
    with TestClient(app) as c:
        r = c.get("/style.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]

def test_script_served():
    with TestClient(app) as c:
        r = c.get("/app.js")
    assert r.status_code == 200
    assert "application/javascript" in r.headers["content-type"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_static.py -v`
Expected: 新增 2 项 FAIL(文件尚不存在)

- [ ] **Step 3: 实现 `static/index.html`(页面骨架)**

DOM 结构:`#messages`(消息区,内含一个欢迎客服气泡)、底部输入区 `#input`(textarea)+ `#send`(按钮);`<link rel="stylesheet" href="/style.css">`、`<script src="/app.js" defer></script>`。

- [ ] **Step 4: 实现 `static/style.css`(暖橙样式)**

`:root` 定义 Global Constraints 的 7 个 CSS 变量;`.bubble.user` 右对齐橙底白字带右下小尾巴、`.bubble.assistant` 左对齐淡橙底深字带头像;顶栏「🤖 在线客服 + 绿点」;`#input` 圆角 + `#send` 橙底 hover 变 `--brand-hover`;流式光标 `.cursor` 用 `@keyframes blink` 闪烁;`body` 暖米白底、聊天卡片居中 max-width 760px、移动端全宽;消息区 `overflow-y:auto`。

- [ ] **Step 5: 实现 `static/app.js`(SSE 解析 + 渲染)**

按 Review Focus 落地四条前端要求:① `send()` 里 `const text = input.value.trim()`,空则 return;`if (state.streaming) return` 并禁用 `#send`。② `fetch("/chat", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({message: text, session_id: state.sessionId})})`;`res.body.getReader()` + `TextDecoder(stream:true)` 逐块读。③ `handleSSE(chunk, bubble)`:`sseBuffer += chunk; const parts = sseBuffer.split("\n\n"); sseBuffer = parts.pop()`;对每个完整帧,逐行剥 `data: ` 前缀后 `JSON.parse`,按 `type` 分发。④ `dispatch(evt, bubble)`:`meta`→`state.sessionId = evt.session_id`;`token`→把 `evt.content` 追加进当前客服气泡并滚到底;`error`→气泡内显示 `evt.message`;`done` 由 finally 收尾(移除光标、`state.streaming=false`、恢复按钮)。发送后新建一个带 `.cursor` 的客服气泡承载流式文本。

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 23 passed(21 旧 + `test_root_serves_index` + `test_style_served` + `test_script_served`)

- [ ] **Step 7: 手动验收(浏览器)**

Run: `.venv/Scripts/python.exe -m uvicorn app.main:app`;浏览器开 `http://127.0.0.1:8000/`。① 发「你好」看回复逐字蹦出(带闪烁光标);② 记下再问「我的订单号是20251115001,货有破损」,然后追问「这个订单想退货怎么办」,确认回复提到订单号(多轮上下文接得住)。Expected:两条都成立。

- [ ] **Step 8: Commit**

```bash
git add static/index.html static/style.css static/app.js tests/test_static.py
git commit -m "feat: chat frontend with SSE streaming + multi-turn"
```

---

## Self-Review 结论

- **Spec 覆盖**:Goals(打开即聊 / SSE 逐字 / 多轮 / 移动端)由 Task 2 覆盖;静态托管由 Task 1 覆盖;Non-goals/Anti-goals 由 Global Constraints 锁定(零依赖、不改业务逻辑)。
- **Review Focus 5 条**:① ② ③ ④ 落在 Task 2 的 `app.js` 实现要求 + 手动验收;⑤ 落在 Task 2 的 pytest 断言。
- **类型一致**:`/chat` 事件字段 `type`/`session_id`/`content`/`message` 与后端 `chat.py` 输出一致;`STATIC_DIR` 在 Task 1 定义、Task 2 复用。
- **比例**:计划与 spec 相当,代码块以断言 + 行为要求为主,无整段实现誊抄。
