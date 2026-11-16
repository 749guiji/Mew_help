# 第三章「Function Calling 工具链」Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统加 LangChain `@tool` 工具链:模型 `bind_tools` 单轮判定工具 → 执行(SSE 推状态帧)→ 回灌收敛逐字流式,聊天落 MySQL,前端气泡带工具徽章。

**Architecture:** FastAPI + SQLAlchemy(同步)+ MySQL(Docker)。5 个 `@tool` 进注册表,`ChatService` 重构成「ainvoke 判定 → 执行器(超时/重试)执行 → ToolMessage 回灌 → base llm astream 收敛」单轮流程。会话上下文内存为主、消息只写落库。

**Tech Stack:** FastAPI 0.163 · SQLAlchemy 2.x + PyMySQL + cryptography · MySQL 8(Docker)· LangChain 1.5 `@tool`/`bind_tools` · pytest(SQLite 内存库)。

**Spec:** `docs/superpowers/specs/2026-11-16-ch03-tool-calling-design.md`

## Global Constraints

- 依赖钉死新增:`sqlalchemy`、`pymysql`、`cryptography`(版本装最新,`requirements.txt` 记录)。
- `.env` 加 `DATABASE_URL=mysql+pymysql://root:123456@127.0.0.1:3306/mew_help`;`Settings.database_url` 字段。
- ORM 模型忠实映射 `db/schema.sql`;ENUM 列用 `Enum(..., native_enum=False)`(存 VARCHAR,便于 SQLite 测试)。
- 单轮工具调用:最终收敛用不带工具的 base llm,杜绝二次工具调用。
- 内存历史只存 user + 最终 assistant 答案;工具调用/结果的消息只在当轮内部,不回放。
- 不改 ch01 `/extract`、不改 ch02 静态托管;前端走 Vibe Coding(无 TDD/review)。
- 模型 `qwen3:8b`;工具选择靠 `@tool` docstring 引导,复用 `SYSTEM_PROMPT`。
- commit message 末尾附 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。

## Review Focus

1. **模型该调不调 / 调错工具** → 标注样例集验证正确率(Task 8),含「邮费→查不到」负例。
2. **工具执行超时/失败** → 执行器重试 + 错误文案回灌给模型,不崩不挂起(Task 4/5)。
3. **工具参数 Schema 不合法**(模型给错类型/缺字段)→ 执行前校验 + 错误回灌(Task 3/4)。
4. **落库失败不阻断回复** → DB 写入 try/except 兜底,流式回复照常(Task 5)。
5. **多轮上下文错乱** → 内存历史只存 user+最终答案,工具消息不回放;session_id 复用不新建 conversation(Task 5)。

## File Structure

- `docker-compose.yml`(mysql:8)
- `app/db/__init__.py`(engine / SessionLocal / Base)、`app/db/models.py`(4 模型)、`app/db/seed.py`(灌 faq)
- `app/tools/__init__.py`(注册表)、`app/tools/business.py`(3 个 mock)、`app/tools/db_tools.py`(query_faq/create_ticket)、`app/tools/executor.py`、`app/tools/context.py`(ContextVar)
- `app/services/chat_service.py`(重构)、`app/main.py`(接线)
- `scripts/eval_tools.py`(标注样例集打分)
- `tests/test_config.py`(改)、`test_db.py`、`test_tools.py`、`test_executor.py`、`test_chat_service.py`(改)、`test_routes.py`(改)

---

### Task 1: 基建(依赖 + 配置 + SQLAlchemy 引擎)

**Files:**
- Modify: `requirements.txt`、`.env`、`.env.example`、`app/core/config.py`
- Create: `docker-compose.yml`、`app/db/__init__.py`
- Test: `tests/test_config.py`(改)、`tests/test_db.py`(新)

**Interfaces:**
- Produces: `Settings.database_url: str`(默认 `mysql+pymysql://root:123456@127.0.0.1:3306/mew_help`);`app.db.engine`(绑定 database_url)、`app.db.SessionLocal`(sessionmaker)、`app.db.Base`(declarative_base)。

- [ ] **Step 1: 写失败测试**

`tests/test_config.py` 追加:

```python
def test_database_url_default():
    from app.core.config import Settings
    s = Settings(_env_file=None)
    assert s.database_url.startswith("mysql+pymysql://")
```

`tests/test_db.py`:

```python
from app.db import Base, SessionLocal, engine

def test_engine_and_session_exist():
    assert engine is not None
    assert SessionLocal is not None
    assert Base.metadata is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_db.py -v`
Expected: FAIL(`Settings` 无 `database_url`、`app.db` 不存在)

- [ ] **Step 3: 实现**

`requirements.txt` 加 `sqlalchemy`、`pymysql`、`cryptography` 并 `pip install`;`config.py` 的 `Settings` 加 `database_url: str`;`app/db/__init__.py` 用 `create_engine(get_settings().database_url)` + `sessionmaker` + `declarative_base()`;`docker-compose.yml` 起 `mysql:8`(env `MYSQL_ROOT_PASSWORD=123456`/`MYSQL_DATABASE=mew_help`、`3306:3306`、volume 持久化、挂 `db/schema.sql` 到 `/docker-entrypoint-initdb.d/`)。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: sqlalchemy engine + config + docker mysql"
```

---

### Task 2: 数据模型 + 种子数据

**Files:**
- Create: `app/db/models.py`、`app/db/seed.py`
- Test: `tests/test_db.py`(扩)

**Interfaces:**
- Consumes: `app.db.Base`(Task 1)。
- Produces: `Conversation(id, user_id, status, created_at, updated_at)`、`Message(id, conversation_id, role, content, tool_calls, tool_call_id, created_at)`、`Faq(id, question, answer, category, created_at, updated_at)`、`Ticket(ticket_no, conversation_id, description, ticket_type, status, created_at)`;`seed_faq(session) -> int` 灌入 N 条 FAQ(含「退货政策」,不含「邮费」)。

- [ ] **Step 1: 写失败测试**

`tests/test_db.py` 追加(用 SQLite 内存库 `create_all` + `seed_faq`):

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base
from app.db.models import Conversation, Message, Faq, Ticket
from app.db.seed import seed_faq

@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()

def test_seed_faq_has_return_policy(session):
    n = seed_faq(session)
    assert n >= 1
    q = session.query(Faq).filter(Faq.question.like("%退货%")).first()
    assert q is not None and q.answer

def test_ticket_uses_ticket_no_pk(session):
    t = Ticket(ticket_no="T20261116001", conversation_id=1, description="x", ticket_type="售后", status="待处理")
    session.add(t); session.commit()
    assert session.get(Ticket, "T20261116001") is t
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db.py -v`
Expected: FAIL(`app.db.models` 不存在)

- [ ] **Step 3: 实现模型 + 种子**

`app/db/models.py` 按 DDL 定义 4 个模型,ENUM 列用 `Enum(..., native_enum=False)`;`ticket_no` 主键。`app/db/seed.py` 的 `seed_faq(session)` 插入若干条 FAQ,其中一条含「退货政策」答案(如「7 天无理由退货…」),**不含**「邮费/运费」关键词。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/db/models.py app/db/seed.py tests/test_db.py && git commit -m "feat: sqlalchemy models + faq seed"
```

---

### Task 3: 5 个 @tool + 注册表 + ContextVar

**Files:**
- Create: `app/tools/__init__.py`、`app/tools/business.py`、`app/tools/db_tools.py`、`app/tools/context.py`
- Test: `tests/test_tools.py`(新)

**Interfaces:**
- Consumes: `SessionLocal`(Task 1)、模型( Task 2)。
- Produces: `query_order(order_id: str) -> str`、`query_product(product_name: str) -> str`、`query_logistics(order_id: str) -> str`(mock 随机数据);`query_faq(keyword: str) -> str`(LIKE 查 faq,无结果返回「未查到」);`create_ticket(description: str, ticket_type: str) -> str`(插 tickets,返回 `T+YYYYMMDDHHMMSS+3 位序号`);`ALL_TOOLS: list[BaseTool]`、`TOOL_REGISTRY: dict[str, BaseTool]`;`current_conversation_id: ContextVar[int|None]`。

- [ ] **Step 1: 写失败测试**

`tests/test_tools.py`(mock 工具不碰 DB;DB 工具 monkeypatch `SessionLocal` 为 SQLite):

```python
from app.tools import ALL_TOOLS, TOOL_REGISTRY

def test_registry_has_five_tools():
    assert set(TOOL_REGISTRY) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
    assert len(ALL_TOOLS) == 5

def test_query_order_returns_str():
    out = TOOL_REGISTRY["query_order"].invoke({"order_id": "1001"})
    assert isinstance(out, str) and "1001" in out

def test_query_faq_no_match():
    out = TOOL_REGISTRY["query_faq"].invoke({"keyword": "邮费"})
    assert "未查到" in out
```

(create_ticket 落库 + ticket_no 格式在 Step 1 的 test 里补:`monkeypatch` SessionLocal → SQLite,断言返回以 `T` 开头、写入了 Ticket。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools.py -v`
Expected: FAIL(`app.tools` 不存在)

- [ ] **Step 3: 实现**

`business.py` 用 `@tool` 定义 3 个 mock(内部 `random` 生成订单/商品/物流文案,含传入 id);`db_tools.py` 用 `@tool` 定义 query_faq(体内 `with SessionLocal() as s` 查 `Faq.question.like(f"%{keyword}%")`,取第一条 answer 或「未查到」)和 create_ticket(读 `current_conversation_id.get()`,生成工单号,插 Ticket);`context.py` 定义 `current_conversation_id`;`__init__.py` 汇总 `ALL_TOOLS`/`TOOL_REGISTRY`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools tests/test_tools.py && git commit -m "feat: 5 @tool business functions + registry"
```

---

### Task 4: 工具执行器(超时/重试/错误)

**Files:**
- Create: `app/tools/executor.py`
- Test: `tests/test_executor.py`(新)

**Interfaces:**
- Consumes: `TOOL_REGISTRY`(Task 3)。
- Produces: `class ToolExecutor`;`__init__(self, registry=None, timeout=5.0, retries=2)`;`async execute(self, name: str, args: dict) -> str`(查表、`asyncio.wait_for` 超时、失败重试 `retries` 次,最终成功返回工具输出字符串,失败返回 `"[工具错误] <msg>"`;**永不 raise**)。

- [ ] **Step 1: 写失败测试**

`tests/test_executor.py`(注入假工具,一个正常、一个首次抛错后成功、一个始终超时):

```python
import asyncio
from app.tools.executor import ToolExecutor

class _Boom:
    name = "boom"
    def __init__(self): self.calls = 0
    async def ainvoke(self, args): self.calls += 1; raise RuntimeError("x")

class _Ok:
    name = "ok"
    async def ainvoke(self, args): return "fine"

def test_execute_success():
    ex = ToolExecutor(registry={"ok": _Ok()})
    assert asyncio.run(ex.execute("ok", {})) == "fine"

def test_execute_retries_then_error_string():
    ex = ToolExecutor(registry={"boom": _Boom()}, retries=1)
    out = asyncio.run(ex.execute("boom", {}))
    assert out.startswith("[工具错误]")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor.py -v`
Expected: FAIL(`app.tools.executor` 不存在)

- [ ] **Step 3: 实现**

`executor.py` 按接口实现:取工具 → `asyncio.wait_for(tool.ainvoke(args), timeout)` → 循环重试;捕获 `TimeoutError`/`Exception`;成功返回 `str(result)`,最终失败返回错误串。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_executor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/executor.py tests/test_executor.py && git commit -m "feat: tool executor with timeout + retry"
```

---

### Task 5: ChatService 单轮工具流程 + SSE tool 帧 + 落库

**Files:**
- Modify: `app/services/chat_service.py`
- Test: `tests/test_chat_service.py`(改)

**Interfaces:**
- Consumes: `TOOL_REGISTRY`/`ALL_TOOLS`(Task 3)、`ToolExecutor`(Task 4)、`SessionLocal`(Task 1)、模型( Task 2)、`current_conversation_id`。
- Produces: `ChatService.__init__(self, llm=None, tools=None, executor=None)`;`stream_chat(self, session_id, message)` 事件序:`meta` → (`tool` running/done/error × N)→ `token` × N → `done` / `error`。流程:`llm.bind_tools(tools).ainvoke(messages)` 判定 → 有 `tool_calls` 则逐工具 yield tool 帧 + 执行 + 回灌 `ToolMessage` → base `llm.astream(messages)` yield token;无 `tool_calls` 则 yield `ai_msg.content` 单 token。落库:首次建 conversation,写 user / assistant 工具调用 / tool 结果 / assistant 最终答案;内存历史只 append user + 最终答案。

- [ ] **Step 1: 写失败测试**

`tests/test_chat_service.py` 追加(FakeLLM 首次 `ainvoke` 返回带 `tool_calls` 的 AIMessage,`astream` 流式最终答案;FakeTool 返回固定串;SessionLocal monkeypatch 成 SQLite):

```python
def test_stream_chat_emits_tool_frame_and_persists():
    ...  # 断言事件序列含 {"type":"tool","name":"query_faq","status":"running"} 与 "done"
    ...  # 断言 messages 表有 user / assistant / tool / assistant 四条
```

(具体 fake 类与断言在实现者写测试时按接口补全;关键断言:tool 帧存在、最终 token 存在、落库 4 条消息。)

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_service.py -v`
Expected: 新增断言 FAIL(旧 `stream_chat` 无 tool 帧/无落库)

- [ ] **Step 3: 实现**

按接口重构 `stream_chat`;`bind_tools` 判定与 `astream` 收敛用同一个 `llm`,收敛用 base llm(不带工具);DB 写入全部 try/except 兜底不阻断回复;`current_conversation_id.set(...)` 在执行工具前设置。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_chat_service.py -v`
Expected: PASS(旧测试仍过 + 新增过)

- [ ] **Step 5: Commit**

```bash
git add app/services/chat_service.py tests/test_chat_service.py && git commit -m "feat: single-turn tool flow + SSE tool frame + persist"
```

---

### Task 6: 组装(main.py 接线)+ 路由

**Files:**
- Modify: `app/main.py`、`app/api/routes/chat.py`(如需要)
- Test: `tests/test_routes.py`(改)

**Interfaces:**
- Consumes: `ChatService`(Task 5)、`ALL_TOOLS`(Task 3)。
- Produces: `create_app(chat_service=None, extraction_service=None)` 默认用 `ChatService(get_llm(), tools=ALL_TOOLS)`;`/chat` 输出含 `tool` 帧。

- [ ] **Step 1: 写失败测试**

`tests/test_routes.py` 追加:用注入的 FakeChatService(带工具流程)断言 `POST /chat` 的 SSE 文本含 `"type":"tool"`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_routes.py -v`
Expected: FAIL

- [ ] **Step 3: 实现**

`main.py` 默认构造带 `tools=ALL_TOOLS` 的 ChatService;确认 `/chat` 不改接口。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 全绿

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_routes.py && git commit -m "feat: wire tools into app"
```

---

### Task 7: 前端工具徽章(Vibe Coding,无 TDD)

**Files:**
- Modify: `static/app.js`、`static/style.css`(如需要)

**Interfaces:**
- Consumes:`/chat` 新增 `{"type":"tool","name":...,"status":"running"|"done"|"error","message":...}` 帧。

- [ ] **Step 1: 直接改(不写失败测试)**

`app.js` 的 `dispatch` 增加 `tool` 分支:收到 running 帧,在客服气泡上方渲染「🔧 <工具名> 执行中…」徽章;done 帧把徽章改为「🔧 <工具名> ✓」;error 帧标红。样式走暖橙。改完浏览器 `Ctrl+F5` 肉眼验收(用户描述效果 → 我迭代)。

- [ ] **Step 2: 语法 + 冒烟**

Run: `node --check static/app.js`;启动服务 curl `/` 确认 200。
Expected: 语法 OK、页面可开

- [ ] **Step 3: Commit**

```bash
git add static/app.js static/style.css && git commit -m "feat: tool trace badge in chat bubble"
```

---

### Task 8: 标注样例集 + 集成验收

**Files:**
- Create: `scripts/eval_tools.py`
- Test: 无(跑脚本看正确率,不套 pytest)

**Interfaces:**
- Consumes: 完整工具链(模型 + 工具 + 执行器)。
- Produces: 打印每条 `(话术, 期望工具, 实际工具)` + 正确率。

- [ ] **Step 1: 写样例集**

`scripts/eval_tools.py` 内置 ~10 条标注:`订单 1001 的物流到哪了 → query_logistics`、`退货政策是什么 → query_faq`、`这个商品 XX 多少钱 → query_product`、`查一下订单 1001 → query_order`、`我要投诉,帮我开个工单 → create_ticket`、`你好 → (无工具)`、`邮费是多少 → query_faq(预期查不到,答「未查到」)` 等。脚本对每条跑 `llm.bind_tools(ALL_TOOLS).ainvoke(...)` 取首个 `tool_call.name`,与标注比对。

- [ ] **Step 2: 起 MySQL + 跑样例集**

Run: `docker compose up -d`;确认库表就绪后 `.venv/Scripts/python.exe scripts/eval_tools.py`
Expected: 正确率 ≥ 70%(至少「物流/退货政策/邮费」三条验收样例正确);「邮费」条选 query_faq 且查无结果。

- [ ] **Step 3: 手动验收 3 条(浏览器)**

起服务,浏览器依次问「订单 1001 的物流到哪了」「退货政策是什么」「邮费是多少」,确认气泡带工具徽章、逐字作答、第三条答「未查到/没有相关信息」。

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_tools.py && git commit -m "feat: tool-selection eval set"
```

---

## Self-Review 结论

- **Spec 覆盖**:基建(Task 1/2)、5 工具(Task 3)、执行器(Task 4)、单轮流程 + 落库(Task 5)、组装(Task 6)、前端徽章(Task 7)、样例集 + 验收(Task 8)——四项目标与验收 3 条均有落点。
- **Review Focus 5 条**:① 落 Task 8 样例集;②③ 落 Task 4 执行器 + Task 5 回灌;④ 落 Task 5 落库兜底;⑤ 落 Task 5 内存历史语义。
- **类型一致**:`TOOL_REGISTRY`/`ALL_TOOLS`(Task 3)→ 执行器(Task 4)→ ChatService(Task 5)→ main(Task 6)同名同形;`ToolExecutor.execute(name, args) -> str`、`stream_chat` 事件名(`meta/tool/token/done/error`)全程一致。
- **比例**:与 spec 相当,代码块以断言 + 签名为主。
