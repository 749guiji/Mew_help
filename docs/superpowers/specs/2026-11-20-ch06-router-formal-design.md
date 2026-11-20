# 第六章「分流器正式版」— 设计文档

日期:2026-11-20
状态:待评审 / 待用户批准

## 1. 意图与动机

ch05 用 LangGraph 搭起了编排骨架,但分流器里三个最关键的部件还是占位:指代消解 `resolve_coref` 原样透传、意图识别 `classify_intent` 是「模型吐类别名 + 子串匹配」、分流是 `route_from_intent` 的七类→四出口硬编码字典。没有 Query 扩写、没有退款/售后子流程、没有槽位处理。

本章把分流器做成正式版——这是整条 Workflow 里最关键、也是容错最敏感的节点:一旦意图判错或指代没补全,后面再好的检索和 Agent 都会答偏。目标不是堆新组件,而是把「理解用户到底在问什么」这件事做扎实。

## 2. 目标 / 非目标 / 反目标

**目标**
1. 指代消解 + Query 改写合并为一次 LLM 调用,结合对话历史把「它能退吗」补全成不依赖上下文的完整问题;问题已完整、指代已明确的原样透传,不强行改写。
2. Query 扩写:把一句问题泛化成多条侧重点不同的检索查询(强制 JSON,字段就 `queries` 一个数组),几条一起检索再去重合并;只对退款退货/售后这类高频、容错低的核心场景做;扩写放在检索侧现查现用,库里知识只留一份。
3. 意图识别走 LLM prompt 路线(prompt 四件套:七类枚举选择题 + 强制 JSON `{intent, confidence}` + 边界 few-shot + 「其他」兜底),不训分类模型。
4. 退款退货/售后走确定性子流程:先拿订单数据 → Query 扩写 → 强制检索政策条款 → 只把「这一单能不能退」交给主力 Agent 判断。
5. 槽位处理:缺订单号不猜,弹订单选择器让用户点选回填;退款原因不追问,提交退款单时从固定类目下拉自选。
6. 前端配套:可点选订单卡片(聊天流内)+ 退款表单(退款原因下拉)。

**非目标**
- 不微调小模型 / 不用 BERT 做意图分类。
- 不做跨会话记忆。
- 不新增组件(不引入新库、新中间件);延用现有聊天上游、Ollama LLM、LangGraph、Milvus、MySQL。

**反目标(就算跑通也算失败)**
- 意图输出 JSON 不稳定、解析靠运气(验收②)。
- 指代消解把已完整的问题强行改写,或该补全的不补全(验收①③)。
- 订单数据被悄悄做成真实订单系统(超出本章范围)。
- 拿不准的怪问题被硬塞进七类业务意图(违背「其他」兜底)。

## 3. 约束

- 技术选型定死:FastAPI / SQLAlchemy / LangChain 1.5 / LangGraph 1.2 / Milvus / pymilvus,不换。
- 无订单/退款数据:现有 `query_order` 是 mock 随机(`app/tools/business.py`),`db/schema.sql` 明写「订单走工具内 mock,不建表」。本章订单数据用**固定种子 mock 列表(进程内常量,不建表)**。
- 无「够格大模型」:`.env` 只有 `model=qwen3:8b`(Ollama 本地)。意图识别默认用 `intent_model`(暂等于 `model`),降级路(小模型→置信度低→大模型重判)把骨架搭好,之后换云端 API 改 config 即可。
- Ollama FC 结构化输出不稳(ch05 已知坑:`with_structured_output` 对单字段 schema 偶发 None)。强制 JSON 走「prompt 强制 + 稳健解析」,不走 function-calling 结构化输出。

## 4. 方案取舍(approaches)

### 4.1 订单数据来源
- **A(选)固定种子 mock 列表**:后端放一个进程内订单常量(约 5 条:订单号/商品/金额/状态),`GET /api/orders` 返回给选择器,`query_order` 改成查这个列表不再随机。跟现有 mock 工具、跟「无新增组件」一致,改动最小。
- B:建真实 orders 表 + 种子数据。更「正式」,但系统里订单/商品/物流全是 mock,单建订单表会把「拿订单数据」变成「建订单系统」,超出本章。**弃**。

### 4.2 强制 JSON 输出机制
- **A(选)prompt 强制 JSON + 稳健解析**:prompt 要求只输出 `{"intent":"...","confidence":0.x}`(意图)或 `{"queries":["...",...]}`(扩写);解析时截取首尾大括号、去 markdown 代码围栏、`json.loads`,失败走兜底(意图→「其他」低置信,扩写→`[原始query]`)。对 qwen3:8b 最稳。
- B:`with_structured_output(method="function_calling")` + 兜底(ch04 `rewrite_query` 路线)。qwen3:8b 对单字段 schema 偶发返回 None、偶发包成 `{"type": ...}`(ch05 已验证),意图路由又容不得 fallback 错路。**弃**。

### 4.3 退款/售后子流程形态
- **A(选)平铺节点序列 + 槽位恢复**:在主图里加 `get_order → expand → retrieve_policy` 三个确定性节点,再接 `agent`。缺订单号时 `get_order` 下发订单列表并暂停(END),用户点选后新一轮带 `order_id` 直接恢复到 `get_order` 续跑。不用嵌套子图,状态和边一眼看穿。
- B:LangGraph 嵌套子图(StateGraph 套 StateGraph)。更「模块化」,但子流程只有三个确定性节点 + 一次暂停,嵌套带来的 state 映射和调试成本不值。**弃**。

### 4.4 退款单落库
- **A(选)最小 `refund_requests` 表** + `POST /api/refunds`(order_id + reason ENUM 固定类目 + status + conversation_id)。字段和退款单一一对应。
- B:复用 tickets(加 ticket_type='退款')。tickets 只有 description + ticket_type,退款原因/订单号塞不进去。**弃**。
- C:纯前端演示不落库。验收没要求落库,但「正式版」里退款单该有落点、便于人工跟进。**弃**(作为备选,若用户想更省可降级)。

## 5. 已定设计(how)

### 5.1 分流映射(七类 + 「其他」,六出口)

| 意图 | 出口 |
|------|------|
| 商品咨询 | `knowledge`(单查询检索 → 闸 → Agent,现状不变)|
| 退款退货 | `refund`(子流程:拿订单→扩写→检索政策→Agent)|
| 售后 | `refund`(同上)|
| 物流 | `business`(Agent 调工具,现状不变)|
| 订单 | `business`(同上)|
| 投诉 | `complaint`(现状不变)|
| 闲聊 | `chitchat`(现状不变)|
| 其他(兜底) | `business`(进 Agent 通用回应,不强制工具;置信度低)|

### 5.2 图结构(在 ch05 图上改)

```
reset → slot ──(本轮带 order_id,恢复子流程)──→ get_order
       └────(普通新问题)──→ coref → intent ─┬→ knowledge → retrieve → gate → agent
                                             ├→ refund    → get_order → expand → retrieve_policy → agent
                                             ├→ business  → agent
                                             ├→ complaint → complaint
                                             └→ chitchat  → chitchat
agent/fallback/complaint/chitchat → log → END
```

新增/改动节点:
- `slot`(新,reset 后):读本轮输入的 `order_id`。有值 → **恢复**:保留 `resolved_query`/`route`/`intent`,写 `order_id`、清 `pending_slot`,直接跳 `get_order`(跳过 coref/intent,避免把「选订单」重新判成「订单」意图)。无值 → **新问题**:清掉 `order_id`/`pending_slot`/`route`/`resolved_query`/`intent`(弃选即丢弃挂起的子流程),走 `coref`。
- `coref`(改):`resolve_coref(query, history)` 用 `state["messages"]` 的历史 + 当前 query,一次 LLM 调用完成指代消解 + 口语归一,输出单句 `resolved_query`;已完整则原样透传。
- `intent`(改):`classify_intent(llm, resolved_query)` 走 5.3 的 prompt,返回 `{intent, confidence}`,写 `intent` + `intent_confidence`。
- `get_order`(新):无 `order_id` → 下发订单列表(`order_list`)+ 一条「请选择订单」消息,设 `pending_slot="order_id"`,保留 `resolved_query`/`route`/`intent`,走 END 等用户点选;有 `order_id` → 取订单数据写 `order_data`,进 `expand`。
- `expand`(新):`expand_queries(llm, resolved_query)` 强制 JSON `{"queries":[...]}`(仅退款/售后进到这里,即自动满足「只对高频场景扩写」)。
- `retrieve_policy`(新):对 `queries` 逐条检索、按 chunk_id 去重合并,写 `retrieval_chunks`/`retrieval_score`;检索走轻量路径 + 强制政策(见 5.6);零命中/低分走 `gate` 兜底(见下)。
- `gate`(复用):`knowledge` 与 `refund` 两条路都过闸——`refund` 零命中或 top 分低于阈值 → `fallback`(转人工),对齐 SYSTEM_PROMPT「证据不足→不得编造→转人工」红线。
- `agent`(改):新增 `route=="refund"` 分支,把 `order_data` + 政策证据拼进上下文,明确「判断这一单能否退款/退货,依据政策条款和订单状态作答」;出最终答案后清 `order_id`/`pending_slot`,使同会话下一次退款提问重新弹选择器。

**子流程上下文(跨暂停保留,评审决议)**:`resolved_query`、`route`、`intent`、`order_id`、`pending_slot` 五字段构成「退款/售后子流程上下文」,由 checkpointer 跨暂停保留。`reset` 只清轮内字段(`retrieval_chunks`/`retrieval_score`/`confidence_pass`/`handoff_suggestions`/`expanded_queries`/`order_data`/`order_list`/`intent_confidence`),不碰子流程上下文;上下文在 `slot` 的新问题分支、以及 `agent` 出最终答案后两处被清,杜绝旧订单复用与跨轮泄漏。

`GraphState` 新增字段:`pending_slot: str | None`、`order_id: str | None`、`order_data: dict | None`、`order_list: list`、`expanded_queries: list[str]`、`intent_confidence: float | None`。

### 5.3 意图识别 prompt(四件套)

- 系统提示:七类意图枚举成选择题 + 边界说明(沿用 ch05 的七类边界文字),加「其他」= 拿不准、怪问题、无法归入七类的兜底。
- 输出契约:只输出 `{"intent": "<七类之一或「其他」>", "confidence": <0.0~1.0>}`,不要多余文字。
- few-shot:边界样例(如「这个能退吗」→ 退款退货、「你们怎么回事」→ 投诉、「今天天气怎么样」→ 其他)。
- 兜底:拿不准就 `其他`,不硬塞业务意图。

### 5.4 指代消解 + 改写 prompt

输入 = 最近 N 轮历史(`state["messages"]`,截断到 token 预算)+ 当前 query。指令:把带指代/省略的query补全成一句不依赖上下文也能读懂的完整中文问句,同时把口语模糊问法归一成标准问法;若问题本身已完整、指代已明确,原样返回,不强行改写。输出单句(非 JSON)。

### 5.5 Query 扩写 prompt

输入 = 一句完整标准问法。指令:把它泛化成 2~4 条侧重点不同的检索查询(政策条款 / 流程步骤 / 条件限制 / 时效运费等角度),强制 JSON,字段就 `queries` 一个字符串数组。解析失败 → `[原始query]`。

### 5.6 「强制检索政策条款」

- 扩写 prompt 明确「面向退款退货政策条款」生成查询。
- **轻量检索路径**:`retrieve_policy` 不逐条走 `default_retrieve`(那会再触发一次 `rewrite_query` LLM 调用);扩写产出的查询已是标准问法,直接对每条 query 调 `hybrid_rerank_retrieve`(跳过 rewrite,仅 `expand_synonyms` 喂 BM25 侧),复用缓存好的 store/embedder/reranker。避免 2~4 次冗余 LLM 改写 + 重排。
- 检索时优先过滤 `content_type == "policy"`(即 `return-policy.md` 的 chunk);若 Milvus schema 未把 content_type 建成可过滤字段(待实现期核实 `app/kb/milvus_store.py`),退化为:检索后在 Python 侧按 `content_type=="policy"` 过滤/加权,再合并进证据。

### 5.7 槽位回填(订单选择器)

- `ChatRequest` 增可选 `order_id: str | None`(点选订单卡片时由前端带上来)。
- 子流程缺订单号 → SSE 下发 `order_select` 事件(携带 `order_list`)+「请选择订单」消息;本轮 END。
- 前端把 `order_list` 渲染成可点订单卡片;点选后发 `{message: "已选择订单 X", order_id: "X"}`。
- `slot` 节点识别到 `order_id` → 恢复子流程续跑(见 5.2)。
- `GET /api/orders` 返回种子订单列表(供选择器 / 未来复用)。

### 5.8 退款单

- 新表 `refund_requests`:`id`、`conversation_id`、`order_id`、`reason`(ENUM:7天无理由 / 质量问题 / 发错货 / 少件 / 与描述不符 / 其他)、`status`(待处理/已处理)、`created_at`。
- 新端点 `POST /api/refunds`(直写表,不经 Agent;同 `/api/tickets` 模式)。
- 前端退款表单:退款原因下拉(固定类目)+ 关联已选订单,提交直调 `/api/refunds`。

### 5.9 模型配置(大模型骨架)

- `Settings` 增:`intent_model`(默认 `model`)、`intent_model_base_url`(默认 `base_url`)、`intent_model_api_key`(默认 `api_key`)、`intent_small_model`(默认空)、`intent_escalate_threshold`(默认 0.6)。
- 意图节点:默认用 `intent_model`(当前=qwen3:8b)。当 `intent_small_model` 非空且 != `intent_model` 时,先用小模型判,`confidence < intent_escalate_threshold` 再用大模型重判一次。当前两者相等/为空 → 单遍,骨架就位、零额外成本,之后填云端 API 即生效。

## 6. 评审决议(builder check,设计者自决)

builder check 提出 5 问,均属技术细节、不改变用户可见行为,设计者自决如下:

1. **暂停期间保留哪些状态**:划「子流程上下文」= `resolved_query`/`route`/`intent`/`order_id`/`pending_slot` 五字段跨暂停保留;`reset` 只清轮内字段。恢复时 `expand` 拿得到 `resolved_query`、`agent` 拿得到 `route=="refund"`。
2. **`order_id`/`pending_slot` 何时清、弃选怎么处理**:两处清——`slot` 新问题分支(弃选即丢弃挂起子流程)、`agent` 出最终答案后(杜绝旧订单复用)。
3. **「其他」/低置信在单模型下怎么走**:「其他」→ 现有带工具 `agent`(不另建无工具 LLM 路,「不强制工具」由 SYSTEM_PROMPT 已有规则承载);`intent_confidence` 只记录日志,`intent_small_model` 未配时升级是 no-op。
4. **`retrieve_policy` 是否跳过逐条 rewrite**:是,走轻量 `hybrid_rerank_retrieve`(见 5.6),省 2~4 次冗余 LLM 改写。
5. **政策零命中/低分**:退款路复用 `gate`,零命中或低于阈值 → `fallback`(转人工),对齐 SYSTEM_PROMPT 红线。

## 7. 验证方式(本章「测试」的映射)

- 可单测任务走 TDD:路由映射、`slot`/`get_order`/`expand`/`retrieve_policy` 节点、图边与暂停恢复、`/api/orders`、`/api/refunds`、`refund_requests` 迁移、JSON 解析器(喂坏 JSON 断言兜底)。
- 非可单测任务(prompt/数据)用标注样例/评估集跑验证替代 TDD:
  - 指代消解+改写:`scripts/eval_coref.py` + `tests/data/coref_samples.json`(多轮指代用例:物流→退款→物流,每轮补全对不对)。
  - 意图识别:`scripts/eval_intent.py` 扩样例到 `tests/data/intent_samples.json`(加边界 + 「其他」怪问题 + 断言 JSON 可解析 + confidence 范围)。
  - Query 扩写:`scripts/eval_expand.py` + `tests/data/expand_samples.json`(断言 JSON 可解析、queries 非空、覆盖关键侧重点)。
- 前端用 Vibe Coding 直接做(不套 brainstorm/TDD/review)。

## 8. 留给实现者(builder 自决)

- 种子订单列表放哪个模块、具体 5 条订单内容。
- 指代消解/意图/扩写各自的 prompt 精确措辞与 few-shot 条数(满足 5.3/5.4/5.5 的约束即可)。
- 历史截断与 token 预算的接线细节(复用 ch05 `_trim_history`)。
- Milvus 侧「policy 过滤」最终取「Milvus 字段过滤」还是「Python 侧 content_type 过滤」(取决于 `milvus_store.py` schema,先查证再定)。
- 退款原因类目文案的最终措辞。
- SSE 事件名与前端帧字段命名(在 5.7/5.8 语义下自定)。
