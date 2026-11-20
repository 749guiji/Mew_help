# 分流器正式版(ch06)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 占位的分流器升级为正式版——指代消解+改写、意图识别(JSON 四件套)、Query 扩写、退款/售后确定性子流程、订单槽位回填、退款单,含前端订单卡片与退款表单。

**Architecture:** 在现有 LangGraph 编排图(`app/graph/`)上,把 `coref`/`intent` 两个节点从占位升级为 LLM prompt + 稳健 JSON 解析;新增 `slot`/`get_order`/`expand`/`retrieve_policy` 四个确定性节点构成退款/售后子流程;缺订单号时子流程暂停、经 SSE 下发订单列表,用户点选后带 `order_id` 恢复续跑。订单数据用进程内 mock 常量(不建表);退款单落新表 `refund_requests`。

**Tech Stack:** FastAPI 0.163 / SQLAlchemy 2.1 / LangChain 1.5 / LangGraph 1.2 / Milvus Lite / Ollama(qwen3:8b)。无新增组件。

**Spec:** `docs/superpowers/specs/2026-11-20-ch06-router-formal-design.md`

## Global Constraints

- 技术选型定死,不换:FastAPI / SQLAlchemy / LangChain 1.5 / LangGraph 1.2 / Milvus / pymilvus。
- 无新增组件、无新增库;延用现有聊天上游、Ollama LLM、LangGraph、Milvus、MySQL。
- 强制 JSON 输出走「prompt 强制 + 稳健解析」,**不用** `with_structured_output`(qwen3:8b FC 不稳,ch05 已知坑)。
- 订单数据 = 进程内 mock 常量列表,不建 orders 表。
- 意图 JSON 字段就 `intent` + `confidence` 两个;扩写 JSON 字段就 `queries` 一个数组。
- 退款原因用 ENUM `native_enum=False`(避免 MySQL ENUM mojibake,ch05 已知)。
- 涉及库 API 先 Context7 查最新文档再动手;本环境 Context7 被墙则 WebSearch 兜底(ch05 记录)。本计划已核对现有代码中 LangGraph 1.2 / Milvus 用法,新接口只在用到时查。
- 前端用 Vibe Coding 直接做,不套 brainstorm/TDD/code-review。
- 非可单测任务(纯 prompt/数据)把 TDD 换成标注样例评估集跑验证;prompt 函数的外壳(JSON 解析/透传/兜底)仍用 FakeLLM 做单测。

## Review Focus

1. **多轮指代切换**:同会话「物流→退款→物流」,每轮 coref 都要用对历史,不能把上一轮「退款」的指代带进新一轮「物流」。→ Task 2 的 eval 样例含此切换场景。
2. **弃选订单**:弹了选择器但用户不点、转问别的 → 挂起的 refund 子流程必须被丢弃,不残留 `order_id`。→ Task 6 `test_slot_resume` 弃选路径。
3. **同会话二次退款**:第一次选过订单后,第二次「这个能退吗」要重新弹选择器,不复用旧订单。→ Task 6 `test_slot_resume` 二次退款场景。
4. **坏 JSON / 模型乱输出**:意图或扩写返回非 JSON → 意图落「其他」(confidence 0)、扩写回退 `[原query]`,不崩。→ Task 3/4 的解析兜底单测。
5. **政策零命中**:退款路检索不到 policy chunk → 走 fallback 转人工,不硬答。→ Task 6 `test_graph` 零命中路由 fallback。

---

### Task 1: 订单数据源(mock 常量 + `GET /api/orders`)

**Files:**
- Create: `app/orders.py`
- Create: `app/api/routes/orders.py`
- Modify: `app/main.py`(注册 orders router)
- Modify: `app/tools/business.py`(`query_order` 改查 mock 列表)
- Test: `tests/test_orders.py`、`tests/test_tools.py`(更新 query_order)

**Interfaces:**
- Produces: `app.orders.SEED_ORDERS: list[dict]`、`list_orders() -> list[dict]`、`get_order(order_id: str) -> dict | None`;`GET /api/orders` 返回 `{"orders": [...]}`。
- Order dict 字段:`{"order_id": str, "product": str, "amount": str, "status": str}`,status 取值集合 `{"已支付，待发货","已发货，运输中","已签收","已完成","退款中"}`(与现有 `query_order` 一致)。

- [ ] **Step 1: 写失败测试** `tests/test_orders.py`

```python
def test_seed_orders_shape():
    from app.orders import list_orders
    orders = list_orders()
    assert len(orders) >= 5
    for o in orders:
        assert {"order_id", "product", "amount", "status"} <= o.keys()

def test_get_order_hit_and_miss():
    from app.orders import get_order
    assert get_order("1001")["product"]
    assert get_order("nope") is None
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_orders.py -v` → FAIL(模块不存在)

- [ ] **Step 3: 实现 `app/orders.py`** `SEED_ORDERS`(5 条:1001 无线蓝牙耳机/1002 保温杯/1003 机械键盘/1004 帆布鞋/1005 充电宝,status 从给定集合取)+ `list_orders()` 返回副本 + `get_order(order_id)` 线性查找。

- [ ] **Step 4: 实现 `GET /api/orders`** `app/api/routes/orders.py`:`@router.get("/api/orders")` 返回 `{"orders": list_orders()}`;在 `app/main.py` `include_router(orders_routes.router)`。加 `tests/test_orders.py::test_orders_endpoint`(用 `TestClient` / 现有 `test_routes.py` 模式)。

- [ ] **Step 5: 改 `query_order`** `app/tools/business.py`:`get_order` 命中返回 `f"订单 {order_id}:{product} 金额{amount} 状态:{status}"`,未命中返回 `f"未查到订单 {order_id}"`;删除 `random` 随机状态。更新 `tests/test_tools.py` 断言不再随机。

- [ ] **Step 6: 跑测试通过** `pytest tests/test_orders.py tests/test_tools.py -v` → PASS

- [ ] **Step 7: Commit** `feat(ch06): 订单 mock 数据源 + GET /api/orders + query_order 查列表`

---

### Task 2: 指代消解 + Query 改写

**Files:**
- Modify: `app/prompts/system.py`(新增 `COREF_REWRITE_SYSTEM_PROMPT`)
- Modify: `app/graph/nodes.py`(重写 `resolve_coref` 签名与实现)
- Test: `tests/test_intent.py`(更新 `resolve_coref` 测试)
- Create: `tests/data/coref_samples.json`、`scripts/eval_coref.py`

**Interfaces:**
- Produces: `resolve_coref(llm, query: str, history: list[str]) -> str`(history 为已提取的最近若干轮用户/助手消息文本;输出单句 resolved 问句,空输出/异常回退 `query`)。
- 后续 Task 6 的 `_coref_node` 消费:`resolve_coref(llm, state["raw_query"], [m.content for m in state["messages"]])`。

- [ ] **Step 1: 写失败测试** `tests/test_intent.py`

```python
def test_resolve_coref_passthrough_when_complete():
    assert resolve_coref(FakeLLM("物流到哪了"), "物流到哪了", []) == "物流到哪了"

def test_resolve_coref_falls_back_on_empty():
    assert resolve_coref(FakeLLM(""), "它能退吗", ["我买了耳机"]) == "它能退吗"

def test_resolve_coref_returns_model_output():
    assert resolve_coref(FakeLLM("耳机能退货吗"), "它能退吗", ["我买了耳机"]) == "耳机能退货吗"
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_intent.py -v` → FAIL(`resolve_coref` 新签名)

- [ ] **Step 3: 写 `COREF_REWRITE_SYSTEM_PROMPT`** `app/prompts/system.py`:指令「结合对话历史,把带指代/省略的 query 补全成一句不依赖上下文也能读懂的完整中文问句,同时口语问法归一成标准问法;若已完整、指代已明确,原样返回不强行改写;只输出这一句,不要解释」。human 模板传入历史 + 当前 query。

- [ ] **Step 4: 实现 `resolve_coref`** `app/graph/nodes.py`:用 `ChatPromptTemplate` 拼历史(截断到 token 预算,复用 ch05 `_trim_history` 思路)+ query,invoke 后 strip 引号/围栏/空白;空串回退 `query`。

- [ ] **Step 5: 造评估集 + eval 脚本** `tests/data/coref_samples.json` 至少 5 条,含**物流→退款→物流切换**场景(如 3 轮历史 + 当前「它能退吗」→ expected 含「耳机」);`scripts/eval_coref.py` 读样例、`get_llm` 真跑、按 `got == expected or all(t in got for t in must_contain)` 判 OK、打印 accuracy。

- [ ] **Step 6: 跑单测通过 + 跑 eval** `pytest tests/test_intent.py -v` → PASS;`PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_coref.py` → 打印逐条 OK/MISS + accuracy。

- [ ] **Step 7: Commit** `feat(ch06): 指代消解+改写(结合历史的 LLM 单句输出 + 评估集)`

---

### Task 3: 意图识别(JSON 四件套 + 升级骨架)

**Files:**
- Create: `app/graph/json_utils.py`(`parse_json_object(text: str) -> dict | None`)
- Modify: `app/prompts/system.py`(重写 `INTENT_SYSTEM_PROMPT` 为 JSON 四件套)
- Modify: `app/core/config.py`(加 `intent_model`/`intent_model_base_url`/`intent_model_api_key`/`intent_small_model`/`intent_escalate_threshold`)
- Modify: `app/graph/nodes.py`(重写 `classify_intent` 返回 `IntentResult`;加 `maybe_escalate`)
- Test: `tests/test_intent.py`(重写意图测试)、`tests/test_json_utils.py`
- Modify: `tests/data/intent_samples.json`(扩到含「其他」+ 边界)、`scripts/eval_intent.py`

**Interfaces:**
- Produces: `parse_json_object(text: str) -> dict | None`;`IntentResult(intent: str, confidence: float)`(dataclass);`classify_intent(llm, query: str) -> IntentResult`;`maybe_escalate(escalate_llm, query: str, result: IntentResult, threshold: float) -> IntentResult`(当 `escalate_llm is not None and result.confidence < threshold` 才用 `escalate_llm` 重判,否则原样返回)。
- Task 6 的 `_intent_node` 消费:先用 `intent_model` 判,若配了 `intent_small_model` 则用 `maybe_escalate` 升级。

- [ ] **Step 1: 写失败测试** `tests/test_json_utils.py`

```python
def test_parse_clean_json():
    assert parse_json_object('{"intent":"退款退货","confidence":0.9}') == {"intent": "退款退货", "confidence": 0.9}

def test_parse_with_markdown_fence():
    assert parse_json_object('```json\n{"queries":["a","b"]}\n```') == {"queries": ["a", "b"]}

def test_parse_garbage_returns_none():
    assert parse_json_object("我不知道") is None
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_json_utils.py -v` → FAIL

- [ ] **Step 3: 实现 `parse_json_object`** `app/graph/json_utils.py`:定位首 `{` 到末 `}`,去 markdown 围栏,`json.loads`,异常返回 `None`。

- [ ] **Step 4: 重写 `INTENT_SYSTEM_PROMPT`** 四件套:①七类枚举成选择题(沿用 ch05 边界文字)+「其他」兜底;②输出契约「只输出 `{"intent":"...","confidence":0.0~1.0}`,不要多余文字」;③边界 few-shot(「这个能退吗」→退款退货、「你们怎么回事」→投诉、「今天天气怎么样」→其他 等);④拿不准归「其他」。

- [ ] **Step 5: 加 config** `app/core/config.py`:`intent_model: str`(默认取 `model`)、`intent_model_base_url`(默认 `base_url`)、`intent_model_api_key`(默认 `api_key`)、`intent_small_model: str | None = None`、`intent_escalate_threshold: float = 0.6`。加 `tests/test_config.py` 断言默认值。

- [ ] **Step 6: 实现 `classify_intent` + `maybe_escalate`** `app/graph/nodes.py`:`classify_intent` 走 JSON prompt + `parse_json_object`,解析失败或字段缺失 → `IntentResult("其他", 0.0)`;`maybe_escalate` 按接口定义。更新 `tests/test_intent.py`:

```python
def test_classify_intent_parses_json():
    r = classify_intent(FakeLLM('{"intent":"退款退货","confidence":0.9}'), "这个能退吗")
    assert r.intent == "退款退货" and r.confidence == 0.9

def test_classify_intent_falls_back_to_other_on_bad_json():
    r = classify_intent(FakeLLM("我不知道"), "这个能退吗")
    assert r.intent == "其他" and r.confidence == 0.0
```

- [ ] **Step 7: 扩评估集 + 更新 eval 脚本** `tests/data/intent_samples.json` 扩到含 7 类 + 「其他」怪问题 + 边界(≥12 条);`scripts/eval_intent.py` 改用 `result.intent` 并打印 confidence。

- [ ] **Step 8: 跑单测通过 + 跑 eval** `pytest tests/test_json_utils.py tests/test_intent.py tests/test_config.py -v` → PASS;`PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_intent.py` → 打印逐条 + accuracy。

- [ ] **Step 9: Commit** `feat(ch06): 意图识别 JSON 四件套 + 其他兜底 + 大模型升级骨架`

---

### Task 4: Query 扩写

**Files:**
- Modify: `app/prompts/system.py`(新增 `EXPAND_SYSTEM_PROMPT`)
- Modify: `app/graph/nodes.py`(加 `expand_queries`)
- Test: `tests/test_expand.py`
- Create: `tests/data/expand_samples.json`、`scripts/eval_expand.py`

**Interfaces:**
- Produces: `expand_queries(llm, query: str) -> list[str]`(强制 JSON `{"queries":[...]}`,解析失败/空 → `[query]`)。
- Task 6 的 `_expand_node` 消费:`expand_queries(llm, state["resolved_query"])`。

- [ ] **Step 1: 写失败测试** `tests/test_expand.py`

```python
def test_expand_parses_queries():
    assert expand_queries(FakeLLM('{"queries":["退款政策","退货条件"]}'), "这个能退吗") == ["退款政策", "退货条件"]

def test_expand_falls_back_on_bad_json():
    assert expand_queries(FakeLLM("不知道"), "这个能退吗") == ["这个能退吗"]

def test_expand_falls_back_on_empty_queries():
    assert expand_queries(FakeLLM('{"queries":[]}'), "这个能退吗") == ["这个能退吗"]
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_expand.py -v` → FAIL

- [ ] **Step 3: 写 `EXPAND_SYSTEM_PROMPT` + 实现 `expand_queries`**:指令「把一句完整问法泛化成 2~4 条侧重点不同的检索查询(政策条款/流程步骤/条件限制/时效运费等角度),强制 JSON,字段就 `queries` 一个字符串数组」;实现走 `parse_json_object`,取 `queries` 若非空 list 返回,否则 `[query]`。

- [ ] **Step 4: 造评估集 + eval 脚本** `tests/data/expand_samples.json`(退款/退货/售后类 query,每条带 `must_cover` 关键侧重点);`scripts/eval_expand.py` 判 JSON 可解析、queries 非空、`must_cover` 全覆盖。

- [ ] **Step 5: 跑单测通过 + 跑 eval** `pytest tests/test_expand.py -v` → PASS;`PYTHONPATH=. .venv/Scripts/python.exe scripts/eval_expand.py` → 打印逐条。

- [ ] **Step 6: Commit** `feat(ch06): Query 扩写(强制 JSON queries[] + 兜底 + 评估集)`

---

### Task 5: 政策多查询检索(`retrieve_multi`)

**Files:**
- Modify: `app/kb/retrieval.py`(加 `retrieve_multi`)
- Test: `tests/test_retrieval.py`(加多查询去重合并测试)

**Interfaces:**
- Produces: `retrieve_multi(queries: list[str], *, content_type: str = "policy") -> list[ScoredChunk]`。复用缓存 `_get_store`/`_get_embedder`/`_get_reranker`;对每条 query 调 `hybrid_rerank_retrieve(query, session=..., store=..., embedder=..., reranker=..., filter=f'content_type == "{content_type}"', sparse_text=expand_synonyms(query))`;按 `chunk_id` 去重合并,`score` 取最高,返回按 score 降序。
- Task 6 的 `_retrieve_policy_node` 消费:`retrieve_multi(state["expanded_queries"])`。

- [ ] **Step 1: 写失败测试** `tests/test_retrieval.py`(monkeypatch 缓存组件,fake store/embedder/reranker 返回带重复 chunk_id 的命中,断言去重后数量、score 取高、filter 含 `content_type == "policy"`)。

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 `retrieve_multi`** 按接口定义;注意 `_escape_filter_string` 已存在,filter 字符串用 `f'content_type == "{content_type}"'`。

- [ ] **Step 4: 跑测试通过** `pytest tests/test_retrieval.py -v` → PASS

- [ ] **Step 5: Commit** `feat(ch06): 政策多查询检索 retrieve_multi(强制 content_type=policy + 去重合并)`

---

### Task 6: 分流器核心(router + state + 图重写 + slot/恢复)

**Files:**
- Modify: `app/graph/router.py`(映射加「其他」+ 退款退货/售后→`refund`)
- Modify: `app/graph/state.py`(加 6 个字段)
- Modify: `app/graph/build.py`(重写 reset/slot/coref/intent + 新增 get_order/expand/retrieve_policy 节点与边)
- Test: `tests/test_router.py`(更新)、`tests/test_graph.py`(更新)、`tests/test_slot_resume.py`(新)

**Interfaces:**
- Consumes: `resolve_coref`/`classify_intent`/`maybe_escalate`/`expand_queries`(Task 2-4)、`retrieve_multi`(Task 5)、`get_order`/`list_orders`(Task 1)、`IntentResult`(Task 3)。
- Produces: 编译后的图 `build_graph(checkpointer=None)`;`GraphState` 新字段 `pending_slot: str | None`、`order_id: str | None`、`order_data: dict | None`、`order_list: list`、`expanded_queries: list[str]`、`intent_confidence: float | None`。

- [ ] **Step 1: 更新 router** `app/graph/router.py`:`_ROUTE_BY_INTENT` = 商品咨询→`knowledge`、退款退货→`refund`、售后→`refund`、物流→`business`、订单→`business`、投诉→`complaint`、闲聊→`chitchat`、其他→`business`。更新 `tests/test_router.py`。

- [ ] **Step 2: 更新 state** `app/graph/state.py` 加 6 字段。更新 `tests/test_graph.py` 现有断言。

- [ ] **Step 3: 重写 `reset`** `app/graph/build.py`:`_reset_node` 只清轮内字段(`retrieval_chunks`/`retrieval_score`/`confidence_pass`/`handoff_suggestions`/`expanded_queries`/`order_data`/`order_list`/`intent_confidence`),**不碰**子流程上下文(`resolved_query`/`route`/`intent`/`order_id`/`pending_slot`)。

- [ ] **Step 4: 新增 `_slot_node`** 读 `state["order_id"]`:有值 → `{"pending_slot": None, "visited": ["slot"]}`(保留 resolved_query/route/intent,跳 `get_order`);无值 → 清 `order_id`/`pending_slot`/`route`/`resolved_query`/`intent`(弃选),跳 `coref`。

- [ ] **Step 5: 升级 `_coref_node`/`_intent_node`** `_coref_node` 传 `[m.content for m in state["messages"]]` 作历史;`_intent_node` 用 `classify_intent`,写 `intent` + `intent_confidence` + `route`;配了 `intent_small_model` 时走 `maybe_escalate`。

- [ ] **Step 6: 新增 `_get_order_node`/`_expand_node`/`_retrieve_policy_node`** `get_order`:无 `order_id` → 设 `pending_slot="order_id"`、`order_list=list_orders()`、返回「请选择订单」消息,走 `wait`(END);有 → 写 `order_data=get_order(order_id)`,走 `expand`。`expand`:`expanded_queries=expand_queries(llm, resolved_query)`。`retrieve_policy`:`retrieval_chunks=retrieve_multi(expanded_queries)`、`retrieval_score=max(score, default=None)`。

- [ ] **Step 7: 重连边** `slot → {get_order | coref}`、`coref → intent`、`intent → {knowledge/business/complaint/chitchat/refund}`、`refund → get_order`、`get_order → {expand | wait(END)}`、`expand → retrieve_policy → gate`、`gate → {agent | fallback}`(knowledge 与 refund 共用 gate)。`_agent_node` 完成 refund 后清 `order_id`/`pending_slot`。

- [ ] **Step 8: 写 `tests/test_slot_resume.py`**(monkeypatch 节点函数为 fake):①不带 order_id 问退款 → 状态 `pending_slot=="order_id"`、`order_list` 非空、答案含「请选择订单」;②下一轮带 `order_id="1001"` → 恢复走完、`expanded_queries` 非空、`retrieval_chunks` 被填;③弃选(不带 order_id 转问物流)→ `order_id` 被清、走 business;④同会话二次退款 → 重新弹选择器(不复用旧 order_id)。更新 `tests/test_graph.py` 覆盖零命中路由 fallback。

- [ ] **Step 9: 跑测试通过** `pytest tests/test_router.py tests/test_graph.py tests/test_slot_resume.py -v` → PASS

- [ ] **Step 10: Commit** `feat(ch06): 分流器核心(六出口 + slot/恢复 + 退款子流程接线)`

---

### Task 7: 主力 Agent refund 分支

**Files:**
- Modify: `app/graph/agent.py`(`react_agent_node` 加 refund 证据注入)
- Test: `tests/test_agent_react.py`(更新)

**Interfaces:**
- Consumes: `format_evidence`(`app/services/generation_quality.py`)、`GraphState.order_data`/`retrieval_chunks`(Task 6)。
- Produces: `react_agent_node` 在 `state["route"] == "refund"` 时,把 `order_data` 单 + `retrieval_chunks` 证据拼进 `HumanMessage`,指令「判断这一单能否退款/退货,依据政策条款和订单状态作答,不得承诺金额/到账时间」。

- [ ] **Step 1: 写失败测试** `tests/test_agent_react.py`:route=refund、带 `order_data`/`retrieval_chunks` 时,断言注入的 HumanMessage 含订单号与证据编号;route=business 时不注入。

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现 refund 分支** `app/graph/agent.py`:在现有「knowledge 注入证据」分支旁加「refund 注入订单 + 证据」分支。

- [ ] **Step 4: 跑测试通过** `pytest tests/test_agent_react.py -v` → PASS

- [ ] **Step 5: Commit** `feat(ch06): 主力 Agent refund 分支注入订单 + 政策证据`

---

### Task 8: chat_service 接线(`order_id` 透传 + `order_select` SSE)

**Files:**
- Modify: `app/schemas/chat.py`(`ChatRequest` 加 `order_id`)
- Modify: `app/api/routes/chat.py`(透传 `payload.order_id`)
- Modify: `app/services/chat_service.py`(`stream_chat` 加 `order_id` 入参、下发 `order_select` 事件)
- Test: `tests/test_chat_service.py`(更新)

**Interfaces:**
- Produces: `stream_chat(session_id, message, order_id: str | None = None)`;图输入 `{"messages": [...], "raw_query": message, "order_id": order_id}`;当 `result.get("order_list")` 非空时 yield `{"type": "order_select", "orders": order_list}`。

- [ ] **Step 1: 写失败测试** `tests/test_chat_service.py`:①`ChatRequest(order_id="1001")` 可解析;②fake graph 返回 `order_list` 时,流里含 `order_select` 事件且 `orders` 非空;③`order_id` 传进图输入。

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现** 三个文件按接口改;`order_select` 事件放在 `done` 前、`handoff` 之后。

- [ ] **Step 4: 跑测试通过** `pytest tests/test_chat_service.py -v` → PASS

- [ ] **Step 5: Commit** `feat(ch06): chat_service 透传 order_id + 下发 order_select 事件`

---

### Task 9: 退款单(`refund_requests` + `POST /api/refunds`)

**Files:**
- Modify: `app/db/models.py`(加 `RefundRequest`)
- Modify: `db/schema.sql`(加 `refund_requests` DDL)
- Create: `app/api/routes/refunds.py`
- Modify: `app/main.py`(注册 refunds router)
- Test: `tests/test_refunds_api.py`

**Interfaces:**
- Produces: `POST /api/refunds`,body `{"session_id": str, "order_id": str, "reason": str}`(reason 枚举 `{"7天无理由","质量问题","发错货","少件","与描述不符","其他"}`),返回 `{"refund_id": int}`。按 `session_id` 解 conversation(复用 `/api/tickets` 模式),写 `RefundRequest`。
- `RefundRequest` 字段:`id`(PK 自增)、`conversation_id`(FK)、`order_id: str`、`reason: Enum(..., native_enum=False)`、`status: Enum("待处理","已处理", native_enum=False)`、`created_at`。

- [ ] **Step 1: 写失败测试** `tests/test_refunds_api.py`(TestClient,复用 `test_tickets_api.py` 模式):提交合法退款 → 返回 `refund_id`;reason 非法 → 422;缺 order_id → 422。

- [ ] **Step 2: 跑测试确认失败** → FAIL

- [ ] **Step 3: 实现** `RefundRequest` 模型 + `db/schema.sql` DDL(带 `SET NAMES utf8mb4`)+ `refunds.py` 路由 + `main.py` 注册。迁移走 `python -m app.jobs.migrate`(幂等)。

- [ ] **Step 4: 跑测试通过** `pytest tests/test_refunds_api.py -v` → PASS

- [ ] **Step 5: Commit** `feat(ch06): 退款单 refund_requests 表 + POST /api/refunds`

---

### Task 10: 前端(Vibe Coding:订单卡片 + 退款表单)

**Files:**
- Modify: `static/app.js`、`static/index.html`、`static/style.css`

**说明**:本章前端走 Vibe Coding——直接按效果迭代,不套 brainstorm/TDD/code-review。以下为需实现的交互,实现细节与视觉按用户反馈即时调整。

- [ ] **Step 1: 订单选择卡片** `dispatch` 处理 `order_select` 事件:在聊天流内渲染可点选的订单卡片列表(订单号 + 商品 + 金额 + 状态);点选后发 `{message: "已选择订单 <id>", order_id: "<id>"}`,清掉卡片、流程接着走。
- [ ] **Step 2: 退款表单** 退款结论后渲染简单表单(关联已选订单号 + 退款原因下拉,类目 `7天无理由/质量问题/发错货/少件/与描述不符/其他`),提交直调 `POST /api/refunds`,成功后回显「已提交退款单」。
- [ ] **Step 3: 样式** 订单卡片与退款表单接入现有 Mew 猫系设计 token(`static/style.css`),与三页视觉一致。
- [ ] **Step 4: 手动验收** `make dev` 起服务,浏览器验证:不带订单号问退款 → 弹卡片 → 点选 → 子流程走完 → 退款表单下拉提交。
- [ ] **Step 5: Commit** `feat(ch06): 前端订单选择卡片 + 退款表单`

---

## 交付核对(对应验收标准)

- 验收① 多轮意图+指代:Task 2 eval(物流→退款→物流)、Task 3 eval(7 类 + 其他)。
- 验收② JSON 稳定 + 其他兜底:Task 3/4 解析兜底单测 + eval。
- 验收③「这个能退吗」先补全再走子流程:Task 6 slot_resume + Task 7 refund 分支。
- 验收④ 不带订单号弹选择器、点选走完:Task 6 slot_resume + Task 8 order_select + Task 10 前端。
- 功能演示:`make dev` 起服务 + `scripts/demo_agent_loop.py`(如需)+ 浏览器走查。
