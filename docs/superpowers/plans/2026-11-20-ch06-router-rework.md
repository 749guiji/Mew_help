# ch06 分流器重构对齐权威方案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native)。Steps use checkbox (`- [ ]`) syntax.

**Goal:** 把 ch06 从「两轮重入 + 6 出口 + 独立 refund 表」重构对齐权威方案:订单选择器改用 LangGraph `interrupt`/`Command(resume)` 原地续跑、分流改 5 出口、退款单复用 tickets、`submit_refund` 工具承载退款两出口、去掉降级运行时。

**Architecture:** 沿用现有 `app/graph/` + `InMemorySaver`(支持 interrupt/resume,进程内续跑满足验收)。图改成权威 5 出口:`escalate`(投诉)/`fallback_script`(闲聊+其他)/`knowledge`(商品咨询)/`refund_flow`(退款退货+售后)/`business`(物流+订单)。`fetch_order` 节点缺单 `interrupt()` 暂停、点选后 `Command(resume=order_id)` 原地续跑。

**Spec(权威):** 桌面 `Mewhelp理论/...# 实战演练...# 功能需求 1.md`(内含权威 Spec + Plan);本仓设计对齐到 `docs/superpowers/specs/2026-11-20-ch06-router-formal-design.md`。

## Global Constraints

- 无新增组件;`interrupt`/`Command` 是 LangGraph 自带能力。checkpointer 用现有 `InMemorySaver`(进程内续跑够用,跨重启持久化非本章目标)。
- 意图识别统一大模型(`settings.intent_model` 种子,空回落 `model`);**不实现降级运行时**(删 `maybe_escalate` 接线)。
- 分流 5 出口;「其他」→ `fallback_script`(script_reply 按意图分文案,零模型调用)。
- 退款子流程只把「这一单能不能退」交主力 Agent;缺单 `interrupt` 不猜不编;退款原因提交时固定类目下拉。
- 退款单复用 `tickets` 表(`ticket_type='退款'`),**删掉 `refund_requests` 表与 `/api/refunds`**。
- Query 扩写严格 3 条 JSON `{"queries":[...]}`;只 refund_flow 检索侧。
- 前端 Vibe Coding;prompt 类产出用评估集替 TDD;确定性代码 TDD。
- interrupt surface(已冒烟钉死):`ainvoke` 返回 `__interrupt__` = `list[Interrupt]`,`[0].value` 取载荷;`Command(resume=v)` 同 thread_id 续跑。

## Review Focus

1. **缺单 interrupt → 点选 resume → query_order 只跑一次**(interrupt 前只读;resume 重跑节点不产生副作用)。
2. **「其他」/闲聊走 script_reply 零模型**(不进 Agent、不花模型调用)。
3. **能退→submit_refund 拦成 refund_form;不能退→直接答不出按钮**。
4. **退款单落 tickets(ticket_type='退款')**,不残留 refund_requests。
5. **interrupt 帧不重复发**(只在缺单那轮发一次;resume 续跑不再发)。

---

### Task 1: 意图识别吃历史 + 降级去运行时

**Files:** `app/graph/nodes.py`(classify_intent 加 history 参、删 maybe_escalate)、`app/graph/build.py`(_intent_node 去 escalate 接线、传 history)、`app/prompts/system.py`、`tests/test_intent.py`、`scripts/eval_intent.py`

- [ ] 写失败测试:`classify_intent(llm, query, history)` 签名 + 传历史进 prompt;`maybe_escalate` 删除后 `_intent_node` 不再调它。
- [ ] 实现:classify_intent 加 history 参数拼进 human;build._intent_node 传历史、删 escalate;删 nodes.maybe_escalate。
- [ ] 跑测试 + eval 通过。

### Task 2: coref prompt 精细 + 扩写严格 3 条

**Files:** `app/prompts/system.py`、`tests/data/coref_samples.json`、`tests/data/expand_samples.json`、`scripts/eval_coref.py`、`scripts/eval_expand.py`

- [ ] coref prompt 加两条规则:订单指代带订单号;问通用政策不绑订单。
- [ ] expand prompt 改「严格 3 条」;expand_queries 取前 3 条非空。
- [ ] eval 样例补订单号/通用政策用例;跑 eval。

### Task 3: list_user_orders(user_id) + order_snapshot

**Files:** `app/orders.py`(改)、`app/tools/business.py`(query_order 用 order_snapshot)、`app/api/routes/orders.py`、`tests/test_orders.py`

- [ ] `order_snapshot(order_id)` 随机种子稳定;`list_user_orders(user_id)` 按 user 稳定 2~4 单;`query_order` 同源。
- [ ] `GET /api/orders?user_id=...` 改按 user 出单。
- [ ] TDD + 回归。

### Task 4: 分流 5 出口 + script_reply + 图重构 interrupt

**Files:** `app/graph/router.py`(5 出口)、`app/graph/state.py`(去 pending_slot/order_list,加 evidence/citations)、`app/graph/build.py`(fetch_order interrupt、retrieve_policy、script_reply、去 slot)、`app/graph/nodes.py`(fetch_order/retrieve_policy/script_reply 节点)、`tests/test_router.py`、`tests/test_graph.py`、`tests/test_slot_resume.py`(改写为 interrupt/resume)

- [ ] router 5 出口:投诉→escalate、闲聊/其他→fallback_script、商品咨询→knowledge、退款退货/售后→refund_flow、物流/订单→business。
- [ ] script_reply 节点(闲聊/其他两套文案)。
- [ ] fetch_order:抽单→缺则 interrupt→拿 order_data;retrieve_policy:扩写3条+去重合并。
- [ ] build.py:5 出口边 + refund_flow 链 + 去 slot;测试改写为 interrupt/resume。

### Task 5: submit_refund 工具 + agent_tools 拦截 + refund 注入

**Files:** `app/tools/business.py`、`app/tools/__init__.py`、`app/graph/agent.py`、`tests/test_agent_react.py`

- [ ] submit_refund 工具;agent_tools 拦成 refund_form 动作(合成 ToolMessage)。
- [ ] _agent_messages refund 路注入 order_data + 政策证据 + 判定指令。

### Task 6: chat_service interrupt + resume 端点 + tickets 复用

**Files:** `app/services/chat_service.py`、`app/api/routes/chat.py`、`app/api/routes/actions.py`(新)、`app/schemas/chat.py`(去 order_id,加 ResumeRequest)、`app/db/models.py`(tickets 枚举加退款)、`db/schema.sql`、`app/api/routes/refunds.py`(删)、`app/main.py`、`tests/`

- [ ] stream_chat:ainvoke 后检 `__interrupt__` → 发 interrupt 事件;删 order_id 透传。
- [ ] `POST /api/actions/resume`(Command(resume) 续跑 SSE);`POST /api/actions/create-refund`(写 tickets ticket_type='退款')。
- [ ] tickets 枚举加「退款」;删 refund_requests 模型/路由/DDL。
- [ ] TDD + 回归。

### Task 7: 前端 interrupt 帧 + resume + refund_form(Vibe Coding)

**Files:** `static/app.js`、`static/index.html`、`static/style.css`

- [ ] 消费 interrupt 事件 → 订单卡片;点选 → `/api/actions/resume` 续流。
- [ ] 消费 refund_form 动作 → 退款表单(原因下拉)→ `/api/actions/create-refund`。

### Task 8: eval 集更新 + 全量回归 + review

- [ ] 扩 eval_intent(8类+其他+多轮漂移)、eval_coref、eval_expand。
- [ ] 全量 pytest 绿;opus 整分支 review;finish 合并推 main。

---

## 交付核对
- 验收4(缺单弹选择器→点选续跑):Task 4+6+7 的 interrupt/resume 全链路。
- 验收2(JSON 稳定 + 其他兜底):Task 1+2 eval。
- 验收1(多轮漂移):Task 1 意图吃历史 + eval。
- 退款单复用 tickets:Task 5+6。
