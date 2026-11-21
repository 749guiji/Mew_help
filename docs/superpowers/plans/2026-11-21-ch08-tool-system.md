# 工具系统(ch08) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把写死的 6 个内置工具升级成「即插即用的工具系统」:统一注册中心 + 统一执行引擎(校验/权限/超时/重试/分诊/格式化/审计)+ 两个自建 MCP Server 接入 + create_ticket 的 interrupt 预览确认流。

**Architecture:** 一个 `ToolRegistry` 存 `RegisteredTool`(name/description/json_schema/read_only/source/impl),内置启动时注册、MCP 每轮 `discover()` 现问现拿合成一份;一个 `ToolExecutor` 作为所有工具调用的统一入口(校验→权限→执行→三分类→格式化→审计);create_ticket 走「agent 记 `pending_ticket` → 专用 `confirm_ticket` 节点 `interrupt` 推预览 → resume 放行或拒绝」。

**Tech Stack:** FastAPI · LangChain 1.5 / langchain-core 1.8 · LangGraph 1.2 · `mcp`(SDK v2,`MCPServer`)· `langchain-mcp-adapters`(`MultiServerMCPClient`)· `jsonschema` · SQLAlchemy · MySQL 8。

**Spec:** `docs/superpowers/specs/2026-11-21-ch08-tool-system-design.md`

## Global Constraints

- MCP SDK **v2**:`from mcp.server import MCPServer`(v1 的 `FastMCP` 旧 import 已删除);`mcp.run(transport="streamable-http", port=...)`。
- `MultiServerMCPClient` 用 `await client.get_tools(server_name=...)`;旧 `async with MultiServerMCPClient(...)` 已删除。MCP 工具 `ainvoke` 返回 `(content, artifact)` 元组(content_and_artifact)。
- MCP 工具名**裸名不加前缀**;审计 `tool_name` 记裸名、`mcp_server` 记来源;`MCP_WRITE_TOOLS` 白名单填裸名。
- 写操作只有 `create_ticket`(内置 `read_only=False`);MCP 默认只读,只有名在 `MCP_WRITE_TOOLS` 才算写。写默认不自动重试(`retry_count` 恒 0)。
- 审计 `log_tool_audit` 内部 try/except 隔离,失败只 log 不抛,绝不拦工具执行。
- LangGraph `interrupt()` resume 时**所在节点从头重跑**——`interrupt` 之前不能有副作用。
- 中文 ENUM 值(审计 status/来源)全库 utf8mb4,`schema.sql` 前 `SET NAMES utf8mb4`。
- `query_faq` 不进 agent(检索由 knowledge 节点做);`query_logistics` 内置下线(由 MCP 接管)。

## Review Focus

1. **MCP 工具与内置同名**:spec 只要求 query_logistics 防重名;未来若有 MCP 暴露与内置同名的工具,agent 的 name→tool 映射二义。→ Task 5 加测试:discover 时跳过与内置同名的 MCP 工具(内置优先)。
2. **MCP 返回非字符串**(content block 列表 / 元组 / ToolMessage):engine 归一化不健壮会把 list 塞给模型。→ Task 3 加测试:`ToolResult.content` 恒为 str。
3. **create_ticket 校验拦下后模型反复重发同一非法调用**:校验错误回灌但模型继续发,不应无限循环。→ Task 6 加测试:非法 create_ticket 不设 `pending_ticket`,循环受 `max_steps` 上限约束。
4. **resume 载荷与当前挂起 interrupt 不匹配**(前端发 `confirm` 但挂的是 select_order):应回错误帧不崩。→ Task 7 加测试。
5. **审计写入失败**:`log_tool_audit` 抛异常时工具执行不能被打断。→ Task 2 加测试。

---

### Task 1: 工具注册中心

**Files:**
- Create: `app/tools/registry.py`
- Modify: `app/tools/__init__.py`
- Test: `tests/test_registry.py`(新建);改 `tests/test_tools.py`(删 query_logistics 断言、`TOOL_REGISTRY`/`ALL_TOOLS` 引用)

**Interfaces:**
- Produces:
  - `RegisteredTool(name: str, description: str, json_schema: dict, read_only: bool, source: str, mcp_server: str | None, impl: Any)` dataclass。
  - `ToolRegistry`:`register(tool: RegisteredTool) -> None`、`get(name) -> RegisteredTool | None`、`builtin() -> list[RegisteredTool]`、`agent_builtin() -> list[RegisteredTool]`(builtin 去掉 `query_faq`)。
  - `json_schema_from_tool(tool) -> dict`:`tool.args_schema.model_json_schema()`(pydantic)。
  - 模块单例 `registry = ToolRegistry()`。

- [ ] **Step 1: 写失败测试** `tests/test_registry.py`

```python
from app.tools.registry import registry, RegisteredTool

def test_agent_builtin_excludes_query_faq_and_logistics():
    names = {t.name for t in registry.agent_builtin()}
    assert "query_order" in names and "create_ticket" in names
    assert "query_faq" not in names and "query_logistics" not in names

def test_create_ticket_is_write():
    t = registry.get("create_ticket")
    assert t.read_only is False and t.source == "builtin"

def test_new_tool_registered_is_picked_up():
    registry.register(RegisteredTool("echo", "desc", {"type": "object"}, True, "builtin", None, _Echo()))
    assert "echo" in {t.name for t in registry.agent_builtin()}
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_registry.py -v` → FAIL(无 registry 模块)
- [ ] **Step 3: 实现 `app/tools/registry.py`**(`RegisteredTool` + `ToolRegistry` + `json_schema_from_tool` + `registry` 单例);重写 `app/tools/__init__.py`:每个内置 `@tool` 包成 `RegisteredTool` 并 `registry.register(...)`;`query_logistics` 不注册;`create_ticket` 用 `read_only=False`;导出 `registry` 与 `current_conversation_id`(删 `ALL_TOOLS`/`TOOL_REGISTRY`)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_registry.py -v` → PASS;同步改 `tests/test_tools.py` 里对 `TOOL_REGISTRY`/`ALL_TOOLS`/`query_logistics` 的引用(其余工具函数测试保留)。
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): 工具注册中心 RegisteredTool + ToolRegistry"`

---

### Task 2: 审计模型 + 落库助手

**Files:**
- Create: `app/tools/audit.py`
- Modify: `app/db/models.py`(加 `ToolAuditLog`)、`db/schema.sql`(加用户给定 DDL)
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: `app.db.Base`、`app.db.SessionLocal`(models 层)。
- Produces: `log_tool_audit(*, conversation_id=None, tool_call_id=None, tool_name, tool_source, mcp_server=None, arguments=None, result_summary=None, status, error_message=None, retry_count=0, duration_ms=None) -> None`;`app.db.models.ToolAuditLog`。

- [ ] **Step 1: 写失败测试** `tests/test_audit.py`

```python
def test_log_audit_writes_row(session):  # session 走 conftest 的 SQLite
    monkeypatch 把 app.tools.audit.SessionLocal 换成 session 工厂
    log_tool_audit(tool_name="query_order", tool_source="builtin", status="成功", retry_count=1, duration_ms=12)
    row = session.query(ToolAuditLog).one()
    assert row.status == "成功" and row.retry_count == 1 and row.tool_source == "builtin"

def test_log_audit_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(audit, "SessionLocal", 抛异常的工厂)
    log_tool_audit(...)  # 不抛,直接返回
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_audit.py -v` → FAIL
- [ ] **Step 3: 实现** `app/db/models.py` 加 `ToolAuditLog`(列与 DDL 一致:`conversation_id` BIGINT NULL、`tool_call_id` VARCHAR64 NULL、`tool_name`、`tool_source` ENUM('builtin','mcp', native_enum=False)、`mcp_server`、`arguments` JSON、`result_summary` TEXT、`status` ENUM('成功','失败','超时','校验拦下','权限拒绝', native_enum=False)、`error_message` VARCHAR512、`retry_count` Integer 默认 0、`duration_ms` Integer NULL、`created_at`);`db/schema.sql` 追加用户给定 `CREATE TABLE tool_audit_logs`(含 `SET NAMES utf8mb4` + 三个索引 + 无外键);`app/tools/audit.py` 实现 `log_tool_audit`(try/except 包住,失败只 `logger.warning`)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_audit.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): tool_audit_logs 模型 + DDL + log_tool_audit"`

---

### Task 3: 统一执行引擎

**Files:**
- Modify: `app/tools/executor.py`(整体重写)、`app/core/config.py`(加 `tool_timeout_seconds`/`tool_retries`/`mcp_write_tools`)、`.env.example`、`requirements.txt`(加 `jsonschema`)
- Test: `tests/test_executor.py`(重写)、`tests/test_config.py`(补新字段)

**Interfaces:**
- Consumes: `RegisteredTool`、`registry.get`、`log_tool_audit`、`app.core.config.get_settings`、`app.tools.context.current_conversation_id`。
- Produces:
  - `ToolResult(kind: str, content: str, draft: dict | None = None)`,`kind ∈ {ok, validation_error, permission_denied, needs_confirmation, error}`。
  - `ToolExecutor(*, timeout: float | None = None, retries: int | None = None, write_allowlist: set[str] | None = None, audit=None)`,默认从 `get_settings()` 取;`audit` 默认 `log_tool_audit`(可注入假函数)。
  - `async execute(self, tool: RegisteredTool, args: dict, *, tool_call_id=None, conversation_id=None) -> ToolResult`
  - `async execute_confirmed(self, tool: RegisteredTool, args: dict, *, tool_call_id=None, conversation_id=None) -> str`(写,无重试,`retry_count` 恒 0)

`execute` 顺序:①`jsonschema.validate(args, tool.json_schema)` 失败→audit「校验拦下」→`validation_error`;②`read_only=False` 且 `source=="mcp"` 且名不在白名单→audit「权限拒绝」→`permission_denied`;③`read_only=False` 且 `source=="builtin"`→`needs_confirmation(draft=args)`(不执行不审计);④读:设 `current_conversation_id` 后 `asyncio.wait_for(tool.impl.ainvoke(args), timeout)`,仅对暂态异常(超时 / MCP 传输·会话类)重试 ≤`retries` 次,正常返回(查到或空)不重试;解包 `(content, artifact)` 元组/块列表→str;审计成功/超时/失败。

- [ ] **Step 1: 写失败测试** `tests/test_executor.py`(重写,用假 `RegisteredTool`)

```python
def _rt(name, read_only=True, json_schema=None, impl=None):
    return RegisteredTool(name, "d", json_schema or {"type": "object", "properties": {}}, read_only, "builtin", None, impl)

def test_validation_error_returns_kind_and_audits(calls):
    # json_schema 要求 required ["order_id"];args={} → ToolResult.kind=="validation_error";audit status=="校验拦下"

def test_read_success_unwraps_tuple_and_audits():
    # impl.ainvoke 返回 ("内容", artifact) → content=="内容";audit status=="成功"

def test_transient_retries_then_recovers():
    # 第一次抛 asyncio.TimeoutError,第二次返回 "ok" → retries=1 → content=="ok";audit retry_count==1

def test_business_empty_no_retry():
    # impl 每次返回 "" (正常空) → 只调用一次,audit status=="成功",不重试

def test_write_confirmed_timeout_no_retry(calls):
    # read_only=False 的 builtin 写工具,impl 慢/超时 → execute_confirmed 抛/返回超时,audit retry_count==0,只调用一次

def test_mcp_write_not_allowlisted_denied():
    # source=="mcp", read_only=False, name 不在白名单 → permission_denied;audit status=="权限拒绝"
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_executor.py -v` → FAIL(旧签名)
- [ ] **Step 3: 实现** 重写 `app/tools/executor.py`;`config.py` 加三字段(`tool_timeout_seconds=120.0`、`tool_retries=2`、`mcp_write_tools: list[str]=[]`);`.env.example` 补三行;`requirements.txt` 加 `jsonschema`。暂态异常类名在实现时从 `langchain_mcp_adapters` / `mcp` 包核实后 import。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_executor.py tests/test_config.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): 统一执行引擎 校验/权限/重试/三分类/审计"`

---

### Task 4: 两个 MCP Server

**Files:**
- Create: `mcp_servers/__init__.py`、`mcp_servers/logistics_server.py`、`mcp_servers/aftersale_server.py`
- Test: `tests/test_mcp_servers.py`(smoke)

**Interfaces:**
- Produces: 两个可独立 `python -m mcp_servers.logistics_server` / `aftersale_server` 起进程的模块;工具名 `query_logistics`(物流)、`check_warranty` + `check_return_progress`(售后);`mcp.run(transport="streamable-http", port=8100/8101)` 在 `if __name__ == "__main__"` 下。

- [ ] **Step 1: 写失败测试** `tests/test_mcp_servers.py`

```python
def test_logistics_has_query_logistics():
    from mcp_servers import logistics_server
    assert logistics_server.mcp.name == "logistics"
    assert "query_logistics" in logistics_server.mcp.list_tools()  # 或等价断言

def test_aftersale_has_two_tools():
    from mcp_servers import aftersale_server
    names = {t.name for t in aftersale_server.mcp.list_tools()}
    assert names == {"check_warranty", "check_return_progress"}

def test_mock_tools_return_str():
    assert isinstance(logistics_server.query_logistics("1001"), str)
    assert isinstance(aftersale_server.check_warranty("1001"), str)
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_mcp_servers.py -v` → FAIL
- [ ] **Step 3: 实现** 两个 server 模块,`@mcp.tool()` 装饰普通函数(mock 随机数据,照第 2 章 `query_logistics` 风格,返回普通字符串);`mcp.run(...)` 收进 `__main__`。工具名/描述/参数 schema 由函数名/docstring/type hints 派生。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_mcp_servers.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): 物流/售后两个 MCP Server (Streamable HTTP)"`

---

### Task 5: MCP Client 接入

**Files:**
- Create: `app/mcp/__init__.py`、`app/mcp/client.py`
- Modify: `app/core/config.py`(加 `mcp_logistics_url`/`mcp_aftersale_url`)、`.env.example`、`requirements.txt`(加 `mcp`、`langchain-mcp-adapters`)
- Test: `tests/test_mcp_client.py`

**Interfaces:**
- Consumes: `RegisteredTool`、`get_settings`。
- Produces: `async def discover_mcp_tools(client=None) -> list[RegisteredTool]`:默认惰性建单例 `MultiServerMCPClient({"logistics": {"transport":"streamable_http","url":settings.mcp_logistics_url}, "aftersale": {...}})`,对每个 server `await client.get_tools(server_name=name)` 包成 `RegisteredTool(source="mcp", mcp_server=name, read_only= name not in write_allowlist, json_schema=归一(args_schema))`;**连不上的 server 记 warn 跳过、不缓存旧工具**;**与内置同名的 MCP 工具跳过(内置优先)**。

- [ ] **Step 1: 写失败测试** `tests/test_mcp_client.py`

```python
class FakeTool:  # 有 .name .description .args_schema( dict 或 pydantic )
class FakeClient:
    async def get_tools(self, server_name=None): ...

def test_discover_tags_source_and_server():
    # fake client 返回 {"logistics": [FakeTool(name="query_logistics")], "aftersale": [...]}
    tools = asyncio.run(discover_mcp_tools(client=FakeClient()))
    assert {t.source for t in tools} == {"mcp"}
    assert {t.mcp_server for t in tools} == {"logistics", "aftersale"}

def test_discover_skips_unreachable_server():
    # FakeClient.get_tools("aftersale") 抛异常 → 仍返回 logistics 的工具,不抛

def test_discover_skips_builtin_name_collision():
    # MCP 提供 name=="query_order" → 结果里没有它(内置优先)
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_mcp_client.py -v` → FAIL
- [ ] **Step 3: 实现** `app/mcp/client.py`;`config.py` 加 `mcp_logistics_url="http://127.0.0.1:8100/mcp"`、`mcp_aftersale_url="http://127.0.0.1:8101/mcp"`;`requirements.txt` 加 `mcp`、`langchain-mcp-adapters`(版本按 Context7 已核 API:SDK v2 `MCPServer`、adapters `get_tools`/`streamable_http`)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_mcp_client.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): MCP Client 现问现拿 discover_mcp_tools"`

---

### Task 6: 意图/路由 + agent 接线

**Files:**
- Modify: `app/prompts/system.py`(`INTENT_SYSTEM_PROMPT` 加「建工单」)、`app/graph/nodes.py`(`INTENTS`/`VALID_INTENTS`)、`app/graph/router.py`、`app/graph/state.py`(加 `pending_ticket`)、`app/graph/build.py`(`_reset_node` 清 `pending_ticket`)、`app/graph/agent.py`(重写 `react_agent_node`)
- Test: `tests/test_agent_react.py`(重写)、`tests/test_router.py`(补建工单)、`scripts/eval_intent.py` + `tests/data/intent_samples.json`(补「建工单」样例)

**Interfaces:**
- Consumes: `registry.agent_builtin()`、`discover_mcp_tools()`、`ToolExecutor.execute`、`ToolResult`、`GraphState.pending_ticket`。
- Produces: `react_agent_node(state, *, llm=None, tools=None, executor=None, max_steps=5, token_budget=None)`——`tools=None` 时从 `registry.agent_builtin() + await discover_mcp_tools()` 拼;对 `create_ticket`(或任何 `needs_confirmation`)记 `pending_ticket={"tool_call_id": tc["id"], "args": result.draft}` 后 break;`submit_refund` 拦截照旧;其余走引擎、ToolMessage 回灌;`intent/路由` 支持「建工单」→ `business`。

- [ ] **Step 1: 写失败测试** `tests/test_agent_react.py`(重写,`FakeExecutor` 返回 `ToolResult`)

```python
def test_read_tool_call_returns_tool_message():
    # FakeLLM 先发 query_order tool_call 再收敛;FakeExecutor 返回 ToolResult("ok","订单已发货")
    # → result["messages"] 含 ToolMessage(content="订单已发货")

def test_create_ticket_valid_sets_pending_and_breaks():
    # FakeLLM 发 create_ticket(description="商品破损", ticket_type="售后")
    # FakeExecutor 返回 ToolResult("needs_confirmation", "", draft={...})
    # → result["pending_ticket"]["args"]["description"]=="商品破损";该轮不追加 ToolMessage

def test_create_ticket_validation_error_feeds_back():
    # FakeExecutor 返回 ToolResult("validation_error", "缺少 description")
    # → result["messages"] 含 ToolMessage(含"缺少 description"),且无 pending_ticket;循环继续到收敛

def test_submit_refund_still_intercepted():
    # submit_refund → result["suggested_actions"][0]["type"]=="refund_form",不调 executor

def test_router_build_order_routes_business():
    from app.graph.router import route_from_intent
    assert route_from_intent("建工单") == "business"
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_agent_react.py tests/test_router.py -v` → FAIL
- [ ] **Step 3: 实现** 改 `system.py`/`nodes.py`/`router.py`/`state.py`/`build.py`/`agent.py`。**非可单测部分**:`INTENT_SYSTEM_PROMPT` 加「建工单」枚举+边界 few-shot(「帮我建个工单」→建工单),在 `tests/data/intent_samples.json` 加 3 条建工单样例,`python scripts/eval_intent.py` 跑通(替代 TDD)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_agent_react.py tests/test_router.py -v` → PASS;`python scripts/eval_intent.py` 建工单样例命中
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): 建工单意图 + agent 动态取工具 + pending_ticket"`

---

### Task 7: confirm_ticket 节点 + interrupt + resume 契约

**Files:**
- Modify: `app/graph/nodes.py`(加 `confirm_ticket`)、`app/graph/build.py`(加节点 + agent 条件边)、`app/services/chat_service.py`(`_stream_invoke` interrupt 帧泛化 + `stream_resume` 载荷)、`app/api/routes/actions.py`(`ResumeRequest` 扩展)
- Test: `tests/test_ticket_confirm.py`(新建)、改 `tests/test_actions.py`、`tests/test_chat_service.py`(resume 载荷)

**Interfaces:**
- Consumes: `ToolExecutor.execute_confirmed`、`log_tool_audit`、`GraphState.pending_ticket`。
- Produces: `confirm_ticket(state) -> dict`(在 `nodes.py`):`interrupt({"type":"ticket_preview","draft":{ticket_type,description}})` → `decision={"confirm": bool}`;确认→`execute_confirmed(create_ticket 工具, args, conversation_id=...)` 落 tickets、审计「成功」、`AIMessage("工单已创建,工单号 {ticket_no}…")`;取消→审计「权限拒绝」、`AIMessage("已取消建工单…")`;均清 `pending_ticket`。
- 图:`agent` →(`pending_ticket` 有值)→ `confirm_ticket` → `log`;否则 `agent` → `log`。
- chat_service interrupt 帧:`kind` 按 payload 分支(`select_order`→`orders`,`ticket_preview`→`draft`);`stream_resume(conversation_id, payload)` 透传,`Command(resume=payload)`。`ResumeRequest` 支持 `order_id` 或 `confirm`。

- [ ] **Step 1: 写失败测试** `tests/test_ticket_confirm.py`(图集成,fake agent 直接置 pending_ticket)

```python
def _build(monkeypatch, pending):
    monkeypatch.setattr(nodes, "classify_intent", lambda llm, q, h="": IntentResult("建工单", 0.9))
    monkeypatch.setattr(nodes, "resolve_coref", lambda llm, q, h: q)
    async def fake_agent(state):
        return {"messages": [("assistant", "")], "pending_ticket": pending, "visited": ["agent"]}
    monkeypatch.setattr(agent, "react_agent_node", fake_agent)
    return build_graph()

def test_interrupt_pushes_ticket_preview(monkeypatch):
    g = _build(monkeypatch, {"tool_call_id": "c1", "args": {"description": "商品破损", "ticket_type": "售后"}})
    r = asyncio.run(g.ainvoke({"messages":[("user","帮我建个工单")], "raw_query":"..."}, config={...}))
    assert r["__interrupt__"][0].value["type"] == "ticket_preview"
    assert r["__interrupt__"][0].value["draft"]["ticket_type"] == "售后"

def test_resume_confirm_writes_ticket_and_audit(monkeypatch, session):
    # SessionLocal monkeypatch 到 SQLite;ainvoke 到 interrupt;再 Command(resume={"confirm":True}) 续跑
    # → tickets 表一条、audit status=="成功"、最终 AIMessage 含 ticket_no

def test_resume_cancel_no_ticket_and_denied_audit(monkeypatch, session):
    # Command(resume={"confirm":False}) → tickets 无新行、audit status=="权限拒绝"、AIMessage 含"取消"

def test_resume_wrong_shape_does_not_crash(monkeypatch, session):
    # interrupt 挂起后 Command(resume="意外的字符串") → confirm_ticket 视为未确认/取消,不抛、tickets 无新行
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_ticket_confirm.py -v` → FAIL
- [ ] **Step 3: 实现** `nodes.confirm_ticket` + `build.py` 节点/边 + `chat_service` 帧/载荷泛化 + `actions.ResumeRequest` 扩展(旧 `order_id` 路径向后兼容)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_ticket_confirm.py tests/test_actions.py tests/test_chat_service.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch08): confirm_ticket 节点 + interrupt 预览 + resume 契约"`

---

### Task 8: 前端工单预览卡(Vibe Coding)

**Files:**
- Modify: `static/app.js`(`dispatch` 遇 `interrupt` 且 `kind=="ticket_preview"` 渲染预览卡 + 确认/取消)、`static/style.css`(预览卡样式)
- Test: `tests/test_static.py`(补 `node --check` 或等价语法断言)

**Interfaces:**
- Consumes: interrupt 帧 `{"type":"interrupt","kind":"ticket_preview","draft":{"ticket_type","description"}}`;`POST /api/actions/resume {conversation_id, confirm: true|false}`。

- [ ] **Step 1:** 按用户描述效果直接做:预览卡显示「工单类型 + 问题描述」+「确认提交」「取消」两按钮;确认→`POST /api/actions/resume {confirm:true}` 并 `streamTo` 渲染续流;取消→`{confirm:false}`。订单选择卡片路径(`kind=="select_order"`)不动。
- [ ] **Step 2:** `node --check static/app.js` 通过;`pytest tests/test_static.py -v` 通过。
- [ ] **Step 3: 提交** `git commit -m "feat(ch08): 前端工单预览卡片(确认/取消)"`

---

### Task 9: 依赖钉版本 + 全量回归 + 迁移验证

**Files:**
- Modify: `requirements.txt`(核对 `mcp` / `langchain-mcp-adapters` / `jsonschema` 与 `langchain-core 1.8` 的兼容版本并钉死)、`.env.example`(补全部新字段)

- [ ] **Step 1:** `pip install -r requirements.txt` 装齐;`python -m app.jobs.migrate` 在存量库幂等建 `tool_audit_logs`(或 `docker compose down -v && docker compose up -d` 全新 init 走 schema.sql)。
- [ ] **Step 2:** 全量 `pytest` 绿(含 Task 1–8 新增/改写测试,预期 ≥ 300)。
- [ ] **Step 3:** 功能演示(验收 1–6 命令序列,写进 dev-notes finish 段):
  - 起两个 MCP Server(两个进程)→ `make dev` → 问「订单 1001 的物流到哪了」经 MCP 查到;
  - Server 侧加一个工具、只重启该 Server → 再问即用;
  - 「帮我建个工单」没讲清 → 追问 → 预览卡 → 确认落 tickets 带工单号 / 取消落「权限拒绝」审计;
  - 人为超时 → 看重试 + 审计重试次数/超时/耗时;写超时 → `retry_count=0`。
- [ ] **Step 4: 提交** `git commit -m "chore(ch08): 钉依赖版本 + 全量回归通过"`
