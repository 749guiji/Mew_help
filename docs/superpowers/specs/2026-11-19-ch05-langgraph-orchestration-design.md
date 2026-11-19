# 设计:ch05 LangGraph 生产级编排(确定性骨架 + 主力 Agent)

日期:2026-11-19
分支:直接 commit 到 `main`(既定惯例)。

## 意图与原因

把客服系统从「单轮 `bind_tools` 工具链」升级成「LangGraph 确定性编排骨架 + 主力 Agent 作为骨架核心节点」:先按规则分流,该检索的强制检索、该直接进 Agent 的直接进,该拒答/安抚的不花模型调用,主力 Agent 只在需要时做 ReAct。本章不追求各节点的正式实现(指代消解、意图识别都先最简),重点是把「编排」这一层立起来。

## 目标 / 非目标 / 反目标

- 目标:① 热身脚本 `scripts/demo_agent_loop.py`(最裸 agent 循环)② LangGraph 图骨架 7 节点 ③ 七类意图四出口(代码硬编码)④ 手写 ReAct 主力 Agent(停止条件 + token 预算 + 追问)⑤ LangGraph State + `InMemorySaver` ⑥ 知识类置信度闸 ⑦ 转人工/建工单双按钮前端。
- 非目标:正式版指代消解/意图识别;上下文管理升级;MCP 接入;数据飞轮入库;真人转人工系统(前端模拟)。
- 反目标:不引入 `create_react_agent`(要手写 ReAct 以精确控制停止条件);不新写业务工具(复用 ch02 五个);不破坏现有 SSE 流式 + MySQL 消息落库 + 前端聊天页。

## 约束

- **LangGraph 1.2.x**(1.0 起无破坏性变更);checkpointer 用内置 `InMemorySaver`(key=`thread_id`=`session_id`)。
- 意图识别 / 指代消解 = 纯 Prompt,非 TDD(用标注样例 eval 代替);分流规则 / 置信度闸 / ReAct = 确定性代码,TDD。
- 模型 `qwen3:8b`(Ollama),FC 偶发返回 None 的已知坑需 `isinstance` 回退。
- 工具复用 ch02:`query_order / query_product / query_logistics`(mock)、`query_faq`(RAG)、`create_ticket`(写 tickets 表)。RAG 检索器复用 ch04 的 `retrieve_with_understanding` + `evidence_coverage`。

## 方案(已选定,备选见下)

### 图骨架与节点

`StateGraph` 七节点线性 + 条件边:

```
START → 指代消解 → 意图识别 → 分流(条件边,四出口)
  ├─ knowledge  → 知识检索 → 置信度闸 → [通过] 主力Agent / [不过] 兜底
  ├─ business   → 主力Agent
  ├─ complaint  → 安抚话术节点(带 handoff 建议)
  └─ chitchat   → 固定话术节点
主力Agent / 兜底 / 安抚 / 固定话术 → 日志记录 → END
```

- **指代消解**:本章原样透传(`resolved_query = raw_query`)。
- **意图识别**:一个 prompt → 结构化 JSON,判 `物流/订单/商品咨询/退款退货/售后/投诉/闲聊` 七类;FC 返回 None 时回退「闲聊」。
- **分流**:纯代码映射表(七类 → `knowledge/business/complaint/chitchat`),写死:
  - 知识类 = 商品咨询、退款退货
  - 业务数据类 = 物流、订单、售后
  - 投诉 → complaint;闲聊 → chitchat
- **知识检索**:仅 knowledge 走;调 `retrieve_with_understanding` 取 Top-K。
- **置信度闸**:仅 knowledge 走;`pass = (max(检索得分) ≥ threshold 且 evidence_coverage ≥ threshold)`,最简版阈值写死在 config;不过则回兜底话术 + 写 `low_confidence_questions`(source 用现有枚举或新增),不进 Agent。
- **主力 Agent**:手写 ReAct 循环节点——循环「LLM 绑工具 → 有 tool_call 就执行并回灌 → 无 tool_call 收敛」;停止条件 = 无 tool_call 或达到 max_steps;token 预算 = 按累计 messages 估算,超限截断;缺必要参数时产出追问话术。业务工具复用 ch02;知识类 Agent 额外带上检索证据作为上下文。
- **日志记录**:末尾节点把本次路径(visited 节点 + route + intent)写进日志(供验收 #1「政策类问题日志能看到检索节点被走到」)。

### State 与 checkpointer

`TypedDict` State:自带 `messages: Annotated[list, add_messages]`(多轮),外加 `raw_query / resolved_query / intent / route / retrieval_chunks / retrieval_score / evidence_coverage / confidence_pass / handoff_suggestions / visited`。`compile(checkpointer=InMemorySaver())`,`thread_id = session_id`。

### 流式

SSE 层用 `graph.astream(..., stream_mode="messages")` 只把**主力 Agent 节点**吐出的 content 逐 token 转发给前端;中间节点的结构化输出不上前端。工具调用以 SSE `tool` 状态帧透传(沿用 ch02 前端工具徽章)。

### 转人工 / 建工单(两件事分开)

后端**只产出建议、不自动执行**:`complaint` 路由(或 Agent 判断合适时)在 state 里放 `handoff_suggestions: ["转人工", "建工单"]`(可一个可两个),随答复下发。前端各渲染独立按钮:
- 「转人工」→ 前端本地显示「已转接人工客服」+「您好,我是客服小猫,请问有什么可以帮您的」,**不调后端**。
- 「建工单」→ 才调 `create_ticket` 写 tickets 表。
- 两个按钮互不绑定;用户不点、继续发消息,当普通对话走。

### 热身脚本

`scripts/demo_agent_loop.py`:不依赖 LangGraph,手写最裸循环(调 LLM → 有 tool_call 执行并喂回 → 无则收敛),演示「Agent 就是带工具的循环」;作为交付物保留。

## 关键裁定(自定,非用户拍板)

- **checkpointer = `InMemorySaver`**(用户已选);消息持久化仍靠现有 MySQL `messages` 表,不重复。
- **ReAct 手写**,不用 `create_react_agent`。
- **意图识别 7 类**用结构化输出 `with_structured_output(method="function_calling")`(沿用 ch03/ch04 的 Ollama 兼容做法),失败回退「闲聊」。
- **置信度闸阈值最简版**写死在 config(默认 `retrieval_min_score=0.5` + `evidence_coverage>=0.5`),正式置信度留给可观测那章。
- **分流规则独立成纯函数模块**(`router.py`),便于 TDD。

## 评审后裁定(builder check 5 问,已定)

1. **政策类归类**:「商品咨询」= 广义知识类,兑住政策/运费/规则(运费怎么算、包邮、发货时效等);「退款退货」兑退货政策;「售后」只指售后进度/工单查询(业务数据类)。意图识别 prompt 据此写边界,保持 7 类。
2. **置信度闸运行时条件**:`pass = (max(检索得分) >= retrieval_min_score)`(证据覆盖度是 ch04 评估指标、运行时无 ground-truth,故不用);不过 → 兜底话术 + `low_confidence_questions`(source=`retrieval_low_conf`)。
3. **建工单后端**:新增 `POST /api/tickets`,前端按钮直调写 tickets 表(description=用户投诉原文、ticket_type=投诉、按 session_id 解出 conversation_id);「转人工」纯前端模拟,不调后端。
4. **保留 citations、去 self_check**:知识类答案仍带 `[n]` 引用并下发 `citations` SSE 帧(前端可点溯源);self_check(「证据够不够」的 LLM 判断)由确定性闸取代,主路径不再调。
5. **Agent 工具清单**:`query_faq` 从 Agent 运行时移除(检索由节点做,避免双重检索);Agent 挂 `query_order / query_logistics / query_product`(业务数据类与退款退货查订单用),建工单场景由前端直调 `/api/tickets` 而非 Agent 调 `create_ticket`。

## 留给 builder 的

- 文件布局、任务拆解、测试策略由 writing-plans 定。粗框:
  - 新增 `app/graph/{state,nodes,router,agent,build}.py` + `scripts/demo_agent_loop.py` + 意图识别 prompt;新增 `app/api/routes/tickets.py`(或并入现有 routes)。
  - 改 `app/services/chat_service.py`(编排替换为图)、`app/prompts/system.py`、`static/app.js`/`index.html`、`requirements.txt`(加 `langgraph`)。
  - 测试:`test_router`(七类分流)、`test_confidence_gate`、`test_agent_react`(fake LLM/工具,验证多步 + 停止条件 + 追问)、`test_graph`(端到端 fake 组件,验证知识类强制检索 + 投诉双建议 + 闲聊/投诉不花模型)、`test_tickets_api`、`test_static`(双按钮存在)、意图识别用标注样例 eval。
