# 第七章「会话上下文管理」— 设计文档

日期:2026-11-20
状态:待评审 / 待用户批准

## 1. 意图与动机

ch06 的 `react_agent_node` 里有一个 `_trim_history`(按字符数简单裁最旧消息),`resolve_coref`/`classify_intent` 共用 `_history_text`(把最近消息拼成字符串)。这是「第一版的简单裁剪」——超长就丢最旧的、丢到哪不知道、也不压缩。本章把它升级成正经的三层上下文管理:近的留原文、中的按规则截短、远的异步压成摘要梗概,预算从模型窗口倒推而不是写死。

只做当前会话内,不跨会话、不存用户画像。

## 2. 目标 / 非目标 / 反目标

**目标**
1. 三层结构 + 两锚点(`summary_upto_msg_id` / `layer1_from_msg_id`),按 MySQL `Message.id` 切段;降级只挪锚点、不搬数据。
2. 摘要走后台异步:层 2 攒到超 token 预算 → 起一次摘要任务 → 压成新段追加进 `conversation_summaries`(只追加不重写)→ `summary_upto` 追到 `layer1_from`。触发看**用量**不数条数;不阻塞当前轮回复。
3. 调模型上下文固定顺序:system(人设/红线/工具定义)→ 层2截短 → 层1原文 → 用户当前句 → 梗概+检索证据合成一条挂在用户句之后(梗概不单独占 system)。
4. token 预算从窗口倒推,不写死常量;中文字数折 token 口径与预算一起标定。
5. conversations/messages 两表 + LangGraph State(messages + add_messages reducer + checkpoint);完整历史与精简版各走各的。
6. 可观测:每轮打 `model_ctx`(Agent)/`history_ctx`(coref/意图);摘要任务触发/开始/完成/跳过/失败全留痕。
7. 前端多会话侧栏 + 两个只读接口。

**非目标**
- 语义检索捞历史、主题重要度留关键事实(理论篇另外两层,本课量级滑窗+摘要够用)。
- 跨会话长期记忆、用户画像。
- 摘要的滚动重写(压完不回头)。

**反目标(跑通也算失败)**
- 摘要生成阻塞当前轮回复(验收 4)。
- 装得下却触发压缩(压缩是成本不是美德,验收 3)。
- 摘要编造对话里没出现的内容、留寒暄。
- 梗概单独占一条 system(前缀缓存作废)。
- 工具结果落进 messages 表(触发口径失真)。

## 3. 约束

- 技术选型定死:LangGraph State(add_messages + checkpoint)、LangChain `trim_messages`、MySQL(conversations/messages)。不换。
- 现有聊天上游不动;消息仍只存 user/assistant 两条(工具结果不落消息表,现状已如此)。
- 涉及 LangChain/LangGraph/SQLAlchemy 具体 API,动手前 Context7 核实(本环境 Context7 被墙则 WebSearch 兜底)。
- 中文字数折 token 的口径是一个标定常数,和预算公式一起用一个测试锁死验收 2 的数字(滑窗 5650 / 层1 3954 / 层2 1695)。

## 4. 方案取舍(approaches)

### 4.1 摘要后台任务机制
- **A(选)独立 asyncio/线程后台,每会话一把去重锁**。`app/core/jobs.py` 是按「作业名」key 的运行器(每名一个线程),不适合「每会话一个摘要任务」。摘要触发点是在某轮上下文拼装时检测到层 2 超预算,`asyncio.create_task` 起后台任务(同进程),失败留痕跳过、下轮再试。不阻塞当前轮。
- B:复用 jobs 运行器。作业名与「会话」不是同一维度,得为每个会话动态注册作业名,状态管理和日志混在 kb 作业里,不合适。**弃**。

### 4.2 token 口径
- **A(选)「中文字数 × 标定常数」折 token**。需求明说「按字数折 token」「口径和预算一起校准,只改一个净效果是反的」,故用字符数代理、常数与预算公式一起标定到验收 2 的数字。不用 tiktoken(真实 tokenizer 会把中文数高、且引入不确定)。
- B:tiktoken 精算。更准但「口径」和需求里的「按字数」不符,且验收数字是按字符口径算出来的。**弃**。

### 4.3 上下文拼装的「完整历史」来源
- **A(选)checkpoint `state.messages` 是完整历史(含工具结果),精简版在调模型前现场拼**。三层切片以 MySQL `Message.id` 锚点定边界,内存里用「user/assistant 消息的序号」把锚点换算成位置;工具消息(无 MySQL 行)归它前面的那条助手轮次的层。这样「完整历史(带工具结果,留 checkpoint)」和「精简版(裁过+带摘要)」各走各的、互不影响。
- B:精简版直接读 MySQL(user/assistant)+摘要,工具结果只在当前轮内存里。缺点:层 2「大块工具结果换一行标识」这条规则落空(历史里没有工具结果)。**弃**。

## 5. 已定设计(how)

### 5.1 数据模型(按用户给的 DDL)

`conversations` 加三列:`summary`(TEXT,投影摘要)、`summary_upto_msg_id`(BIGINT,摘要覆盖到哪条)、`layer1_from_msg_id`(BIGINT,层1起点)。新建 `conversation_summaries`(id/conversation_id/seq/from_msg_id/upto_msg_id/content/created_at,UNIQUE(conv,seq))。

层边界(闭区间,按 `Message.id`):
- `id ≤ summary_upto_msg_id` → 层 3(已摘要,看 `conversations.summary` 投影)。
- `summary_upto < id ≤ layer1_from` → 层 2(截短渲染)。
- `id > layer1_from` → 层 1(原文)。

### 5.2 上下文拼装(每轮调模型前)

产出两种精简版:
- **主力 Agent** 的 `model_ctx`:system(人设/红线/工具定义,每轮一样)→ 层2截短 → 层1原文 → 用户当前句 → 一条 HumanMessage 挂「摘要投影 + 检索证据」。日志原样打摘要全文 + 滑窗逐条 + 窗口条数 + tokens 估算。
- **coref/intent** 共用的 `history_ctx`(字符串):摘要行 + 滑窗(最近 N 轮原文),每轮必打(闲聊兜底轮也打)。

层 2 截短规则:用户原话不动;客服答复留开头 ~40 字;工具结果(仅 checkpoint 里有)换一行 `[工具:query_order(订单 1001)]` 标识。

### 5.3 预算(从窗口倒推)

`历史预算 = 窗口 - 系统提示 - 检索证据预留 - 注入摘要预留 - MAX_OUTPUT_TOKENS - MAX_USER_INPUT_TOKENS - 安全余量 - MAX_AGENT_STEPS×TOOL_RESULT_MAX_TOKENS`。历史预算再取 `min(想留轮数×每轮稳态, 上式)`;层1 七成、层2 三成。

配置新增:`MODEL_CONTEXT_WINDOW`/`MAX_OUTPUT_TOKENS`/`MAX_USER_INPUT_TOKENS`/`MAX_AGENT_STEPS`/`TOOL_RESULT_MAX_TOKENS`/`RERANK_TOP_K`/`CHARS_PER_TOKEN`(标定常数)/`SUMMARY_TARGET_CHARS`。**预算为「字数」口径(评审决议 4)**,`MODEL_CONTEXT_WINDOW` 默认反映 qwen3:8b 的 num_ctx(32768)。启动自检:历史预算 < 一轮稳态就 log 报警。

**conversation_id 进 State(评审决议 3)**:`GraphState` 加 `conversation_id`(+ 现有 `user_id`),`stream_chat` 注入;拼装节点靠 conversation_id 从 MySQL 取 summary/锚点。resume(Command)经 checkpoint 看到同一值。

### 5.4 摘要任务(后台)

- 触发:某轮拼装时层 2 用量 > 层 2 预算 → `asyncio.create_task`(每会话一把去重锁,已在跑则跳过)。
- 执行:读层 2 的 user/assistant 消息 → 用主模型按摘要 prompt 提炼事实与诉求(问过的商品/报过的订单号手机号/明确诉求/未解决),不编、不留寒暄,几十到一两百字 → 追加 `conversation_summaries`(seq+1)→ 更新 `conversations.summary`(投影)+ `summary_upto_msg_id = layer1_from_msg_id`。
- 留痕:trigger/start/done/skip/fail 全打 log,带覆盖边界 [from,upto] + 耗时。

### 5.5 降级两条路径

- 层 1 超预算 → `layer1_from_msg_id` 上移一批(层1最老一批降级成层2)。
- 层 2 超预算 → 起摘要(5.4)。

### 5.6 用户身份 + 前端 + 只读接口

- **用户身份(评审决议 A)**:客户端 `localStorage` 生成稳定 `user_id`,每次请求带上;服务端按 user_id 建/查会话,`conversation_id`(会话)与 `user_id`(身份)解耦,`thread_id = conversation_id`(每会话一个 checkpoint)。无登录。`tickets`/`actions`/`orders` 里按 `user_id==session_id` 的查询改成按 conversation_id 对齐。
- **重进图**:切会话/重启后 checkpoint 空 → 从 MySQL 回灌该会话的 user/assistant 历史再进图;工具结果只活在进程内 checkpoint(重启即失,契合「工具结果不落消息表」)。
- `GET /api/conversations?user_id=...` → 该用户全部会话(新在前、首问预览、已摘要标记)。
- `GET /api/conversations/{id}/messages` → 该会话历史原文。
- 前端聊天页加会话侧栏(加载失败静默降级不影响聊天),点选切换回载历史、能接着聊;「新对话」只开新会话。

## 6. 留给实现者(builder 自决)

- 内存里「MySQL msg_id 锚点 → checkpoint 位置」的精确换算与缓存策略(每轮一次 COUNT 查询即可)。
- 层 2 截短的具体字数(客服答复留 ~40 字)、工具结果标识的精确格式。
- 预算公式各固定开销的具体标定值(以验收 2 数字锁死为验收)。
- 摘要 prompt 精确措辞(满足 5.4 约束)。
- 后台任务用 asyncio 还是 daemon 线程(需与 SQLAlchemy 会话隔离)。
