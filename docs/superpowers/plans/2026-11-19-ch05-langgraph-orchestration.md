# ch05 LangGraph 生产级编排 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把客服从单轮 `bind_tools` 工具链升级成 LangGraph 确定性编排骨架(指代消解→意图识别→分流→知识检索→置信度闸→主力 Agent→日志)+ 手写 ReAct 主力 Agent + 转人工/建工单双按钮前端。

**Architecture:** `StateGraph` 七节点线性 + 条件边四出口;`InMemorySaver` checkpointer(`thread_id=session_id`);主力 Agent 手写 ReAct 循环;SSE 层用 `graph.astream(stream_mode="messages")` 只透传 Agent 节点的 token。

**Tech Stack:** FastAPI + LangGraph 1.2.x + LangChain + qwen3:8b(Ollama)+ SQLAlchemy/MySQL + 原生 JS 前端。

**Spec:** `docs/superpowers/specs/2026-11-19-ch05-langgraph-orchestration-design.md`

## Global Constraints

- **LangGraph 1.2.x**,`InMemorySaver` checkpointer(key=`thread_id`=`session_id`),`StateGraph/START/END` + `add_messages` from `langgraph.graph.message`。
- 意图识别 / 指代消解 = 纯 Prompt,非 TDD(标注样例 eval 代替);分流规则 / 置信度闸 / ReAct = 确定性代码,TDD。
- 工具复用 ch02(`query_order/query_product/query_logistics/query_faq/create_ticket`);`query_faq` 不进 Agent 工具(检索由节点做)。
- `qwen3:8b` FC 偶发返回 None → 结构化调用统一 `isinstance` 回退(意图识别回退「闲聊」)。
- 保留 citations SSE + 前端可点引用;**去 self_check**(被确定性闸取代)。
- 每任务结束提交一次。

## Review Focus

1. **闲聊/投诉不花模型调用**:闲聊/投诉路由必须不触发 LLM。→ Task 6 测试。
2. **知识类强制检索**:知识类(商品咨询/退款退货)必须走检索节点 + 闸,业务数据类不走。→ Task 6 测试。
3. **置信度闸不过回兜底**:检索得分低于阈值 → 兜底话术 + 记 `low_confidence_questions`,不进 Agent。→ Task 4/6 测试。
4. **ReAct 停止条件**:无 tool_call 或 max_steps 收敛;不会无限循环。→ Task 5 测试。
5. **转人工/建工单互不绑定、后端不自动执行**:complaint 只下建议;建工单走独立端点;转人工纯前端。→ Task 8 测试。

---

### Task 1: 基建 + 热身脚本

**Files:** Modify `requirements.txt`(加 `langgraph`);Create `scripts/demo_agent_loop.py`

- [ ] **Step 1: 加依赖** — `requirements.txt` 加 `langgraph==1.2.*`(或 `>=1.2,<2`);`.venv/Scripts/python.exe -m pip install langgraph` 确认可 import。
- [ ] **Step 2: 写热身脚本** — `scripts/demo_agent_loop.py`:不依赖 LangGraph,手写最裸循环(调 `get_llm()` → 有 tool_call 就 `ToolExecutor.execute` 喂回 → 无则打印收敛答案);用 `query_order/query_logistics` 两个工具演示。
- [ ] **Step 3: 验证** — `PYTHONPATH=. .venv/Scripts/python.exe scripts/demo_agent_loop.py "订单 1001 的物流到哪了"` 能跑通(需要 Ollama)。
- [ ] **Step 4: Commit** — `git add requirements.txt scripts/demo_agent_loop.py && git commit -m "feat(ch05): 加 langgraph 依赖 + 手写最裸 agent 循环热身脚本"`

---

### Task 2: State + 分流规则(纯函数,TDD)

**Files:** Create `app/graph/__init__.py`, `app/graph/state.py`, `app/graph/router.py`;Test `tests/test_router.py`

**Interfaces:**
- Produces `GraphState(TypedDict, total=False)`:`messages: Annotated[list, add_messages]`、`raw_query/resolved_query: str`、`intent: str`、`route: str`、`retrieval_chunks: list`、`retrieval_score: float|None`、`confidence_pass: bool|None`、`handoff_suggestions: list[str]`、`visited: list[str]`。
- Produces `route_from_intent(intent: str) -> str`(七类 → `knowledge/business/complaint/chitchat`)。

- [ ] **Step 1: 写失败测试** — `tests/test_router.py`:
  - `test_knowledge_intents`:`商品咨询`/`退款退货` → `knowledge`。
  - `test_business_intents`:`物流`/`订单`/`售后` → `business`。
  - `test_complaint_chitchat`:`投诉`→`complaint`、`闲聊`→`chitchat`。
  - `test_unknown_intent_raises`:未知意图 → `ValueError`。

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_router.py -v`(ModuleNotFoundError)

- [ ] **Step 3: 实现 state.py + router.py** — State 用 `TypedDict` + `Annotated[list, add_messages]`;`route_from_intent` 用 dict 映射,未知 raise。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/graph tests/test_router.py && git commit -m "feat(ch05): GraphState + 七类意图四出口分流规则"`

---

### Task 3: 意图识别(纯 Prompt,标注样例 eval)

**Files:** Modify `app/prompts/system.py`(加 `INTENT_SYSTEM_PROMPT`);Create `app/graph/nodes.py`(指代消解透传 + `classify_intent`);Test `tests/test_intent.py`(fake LLM)+ `tests/data/intent_samples.json`(标注样例)

**Interfaces:**
- Produces `IntentResult(BaseModel)`(`intent: str`);`classify_intent(llm, query: str) -> str`(FC 失败回退 `"闲聊"`);`resolve_coref(query: str) -> str`(原样透传)。

- [ ] **Step 1: 写失败测试** — `tests/test_intent.py`(复用 `FakeLLM` 模式):
  - `test_classify_intent_passthrough`:fake LLM 返回 `IntentResult(intent="订单")` → `"订单"`,且 `called_method == "function_calling"`。
  - `test_classify_intent_falls_back_on_none`:fake LLM 返回 `None` → `"闲聊"`。
  - `test_resolve_coref_passthrough`:`resolve_coref("它到哪了") == "它到哪了"`。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 prompt + classify_intent + resolve_coref** — `INTENT_SYSTEM_PROMPT` 写清 7 类边界(重点:「商品咨询」兼政策/运费/规则;「售后」只指售后进度/工单查询);`classify_intent` 用 `with_structured_output(IntentResult, method="function_calling")`。

- [ ] **Step 4: 运行确认通过** → PASS;再跑标注样例 `tests/data/intent_samples.json`(脚本 `scripts/eval_intent.py` 或直接 pytest 参数化)验证「运费怎么算→商品咨询、我要投诉→投诉、你好→闲聊」等。

- [ ] **Step 5: Commit** — `git add app/prompts/system.py app/graph/nodes.py tests/test_intent.py tests/data/intent_samples.json && git commit -m "feat(ch05): 意图识别(7 类 JSON)+ 指代消解透传"`

---

### Task 4: 知识检索 + 置信度闸(TDD)

**Files:** Modify `app/graph/nodes.py`;Test `tests/test_confidence_gate.py`

**Interfaces:**
- Produces `gate_pass(score: float | None, threshold: float) -> bool`(纯函数);`retrieve_node(state) -> dict`(调 `retrieve_with_understanding` 填 `retrieval_chunks/retrieval_score`)。

- [ ] **Step 1: 写失败测试** — `tests/test_confidence_gate.py`:
  - `test_gate_pass_above_threshold`:`gate_pass(0.6, 0.5) is True`。
  - `test_gate_fail_below`:`gate_pass(0.4, 0.5) is False`;`gate_pass(None, 0.5) is False`。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 gate_pass + retrieve_node** — `retrieve_node` 用 `retrieve_with_understanding`(复用 ch04 缓存组件),取 `max(c.score)` 填 `retrieval_score`,空结果 score=None。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/graph/nodes.py tests/test_confidence_gate.py && git commit -m "feat(ch05): 知识检索节点 + 置信度闸纯函数"`

---

### Task 5: 主力 Agent ReAct 循环(TDD,fake LLM/工具)

**Files:** Create `app/graph/agent.py`;Test `tests/test_agent_react.py`

**Interfaces:**
- Consumes:`ToolExecutor`(ch02)、`get_llm`、`ALL_TOOLS` 去掉 `query_faq` 后的 `AGENT_TOOLS`。
- Produces `AGENT_TOOLS: list`;`react_agent_node(state: GraphState) -> dict`(返回 `{"messages": [...], "visited": [...]}` 增量)。

- [ ] **Step 1: 写失败测试** — `tests/test_agent_react.py`(fake LLM:第一次返回 tool_call、第二次返回 content;fake 工具):
  - `test_single_tool_call_converges`:一轮 tool_call 后收敛,`messages` 含 ToolMessage。
  - `test_multi_step`:两次 tool_call 收敛,中间结果喂回(断言 LLM 被调 ≥3 次)。
  - `test_stop_on_max_steps`:LLM 一直返回 tool_call → 达到 max_steps 停止(不无限循环)。
  - `test_missing_info_asks_user`:LLM 无 tool_call、返回追问话术 → 收敛。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 `react_agent_node`** — 手写循环:`llm.bind_tools(AGENT_TOOLS)` → 有 tool_call 执行(`ToolExecutor`)喂 ToolMessage → 无 tool_call 收敛;`max_steps`(默认 5)+ token 预算(累计消息估算,超限截断);知识类带上 `retrieval_chunks` 格式化成带 `[n]` 的证据注入上下文。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/graph/agent.py tests/test_agent_react.py && git commit -m "feat(ch05): 手写 ReAct 主力 Agent(停止条件/token 预算/追问)"`

---

### Task 6: 图组装 + 安抚/闲聊/日志节点 + checkpointer(TDD)

**Files:** Create `app/graph/build.py`;Modify `app/graph/nodes.py`(安抚/闲聊/日志节点);Test `tests/test_graph.py`

**Interfaces:**
- Produces `build_graph(checkpointer=None) -> CompiledGraph`;`run_graph(query, session_id, ...) -> AsyncIterator[SSE事件]`(流式封装,或留在 chat_service)。

- [ ] **Step 1: 写失败测试** — `tests/test_graph.py`(fake 意图/检索/Agent 节点,验证路由):
  - `test_knowledge_route_forces_retrieval`:知识类 → visited 含 `retrieve` + `confidence_gate`。
  - `test_business_route_skips_retrieval`:业务类 → visited 不含 `retrieve`。
  - `test_complaint_returns_handoff_suggestions`:`投诉` → `handoff_suggestions == ["转人工","建工单"]` 且不调 LLM(Agent 节点未执行)。
  - `test_chitchat_fixed_reply`:`闲聊` → 固定话术、不调 LLM。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现** — `nodes.py` 加安抚/闲聊/日志节点;`build.py` 用 `StateGraph(GraphState)` 组装:START→指代消解→意图识别→`add_conditional_edges`(按 route)→ 各出口 → 日志 → END;`compile(checkpointer=checkpointer or InMemorySaver())`。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/graph/build.py app/graph/nodes.py tests/test_graph.py && git commit -m "feat(ch05): StateGraph 组装 + InMemorySaver + 安抚/闲聊/日志节点"`

---

### Task 7: chat_service 编排替换 + SSE handoff 事件(TDD)

**Files:** Modify `app/services/chat_service.py`;Test `tests/test_chat_service.py`(扩)

**Interfaces:**
- Produces:SSE 事件流新增 `{"type": "handoff", "suggestions": [...]}`;`stream_chat` 内部改为 `graph.astream(..., stream_mode="messages")` + `config={"configurable":{"thread_id": session_id}}`。

- [ ] **Step 1: 写失败测试** — `tests/test_chat_service.py`(fake graph/组件):complaint 路由的流里出现 `handoff` 帧;知识类流里出现 `citations` 帧 + `token` 帧。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现** — 替换 `stream_chat` 内部编排;用 `astream(stream_mode="messages")` 只转发 Agent 节点的 token 为 `token` 帧;工具调用透传 `tool` 帧;`handoff_suggestions` 透传 `handoff` 帧;`retrieval_chunks` 透传 `citations` 帧。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/services/chat_service.py tests/test_chat_service.py && git commit -m "feat(ch05): chat_service 换 LangGraph 编排 + SSE handoff/citations"`

---

### Task 8: 建工单 API + 前端双按钮(TDD)

**Files:** Create `app/api/routes/tickets.py`;Modify `app/main.py`、`static/app.js`、`static/index.html`;Test `tests/test_tickets_api.py`(新)、`tests/test_static.py`(扩)

**Interfaces:**
- Produces `POST /api/tickets`(body `{session_id, description, ticket_type}`)→ 写 tickets 表(按 session_id 解 conversation_id)。

- [ ] **Step 1: 写失败测试** — `tests/test_tickets_api.py`:`test_create_ticket_writes_row`(monkeypatch SessionLocal 用 sqlite,断言 tickets 表新增一行);`test_tickets_rejects_blank_description`。`test_static.py` 扩:index.html 含 `handoff` 容器/按钮标记。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现** — `tickets.py` 端点(复用 ch02 create_ticket 的写库逻辑,但 description/ticket_type 由请求体给、conversation_id 由 session_id 解);前端 `dispatch` 处理 `handoff` 帧渲染两个独立按钮:「转人工」本地显示「已转接人工客服」+「您好,我是客服小猫…」;「建工单」`fetch POST /api/tickets`;互不绑定。

- [ ] **Step 4: 运行确认通过** → `pytest tests/test_tickets_api.py tests/test_static.py -v` PASS;`node --check static/app.js` 干净。

- [ ] **Step 5: Commit** — `git add app/api/routes/tickets.py app/main.py static/app.js static/index.html tests/test_tickets_api.py tests/test_static.py && git commit -m "feat(ch05): 建工单 API + 转人工/建工单双按钮前端"`

---

### 最终验证

- `pytest -q` 全绿;`node --check static/app.js` 干净。
- 起服务后浏览器走验收五条:政策类检索日志、订单物流工具、投诉双按钮、闲聊固定话术、ReAct 多步。
- `scripts/demo_agent_loop.py` 能跑。
