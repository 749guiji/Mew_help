# 第三章「Function Calling 工具链」— 设计

日期:2026-11-16

## Intent / Why

给现有客服系统(ch01 纯对话 + ch02 聊天前端)装上「查数据」能力:用 LangChain `@tool` 定义 5 个业务工具,`bind_tools` 让模型自己决定调不调、调哪个,单轮调用后收敛,最终答案仍走 SSE 逐字流式。用户在聊天页问一句就触发,气泡带「工具轨迹徽章」,聊天记录落库。

## Goals

- 基建:FastAPI + SQLAlchemy 分层骨架,Docker 起 MySQL,建 4 张表并灌测试数据。
- 5 个工具:query_order / query_product / query_logistics(mock,内部随机数据)、query_faq(LIKE 查 faq)、create_ticket(写 tickets)。
- 工具基础设施:注册管理、参数 Schema 校验、执行错误处理、超时重试,结果回灌给模型组织回答。
- 接入现有 `/chat` SSE:模型判定工具 → 执行(先推状态帧)→ 回灌收敛 → 逐 token 流式;聊天(含工具调用与结果)落 conversations/messages;前端气泡显示工具徽章。
- 单轮调用:模型调一次工具就收敛,不做循环。

## Non-goals

- 多轮自动循环的 Agent Loop。
- 向量检索 / RAG。
- 商品/订单/物流接真实第三方 API(演示用 mock)。

## Anti-goals

- 不改 ch01 的 `/extract` 结构化提取,不改 ch02 的静态托管方式。
- 不把工具 Schema 里的内部参数(如 conversation_id)暴露给模型。

## Constraints

- 技术栈定死:FastAPI + SQLAlchemy + MySQL(Docker)+ LangChain `@tool`。
- 模型固定 `qwen3:8b`(Ollama,`.env` 已有)。
- 单轮工具调用,无 Agent Loop。
- Docker Desktop 已启动;MySQL root 密码 `123456`。
- 中文枚举值:工单类型 `售后/投诉/咨询`、工单状态 `待处理/已处理`、会话状态 `进行中/已转人工/已结束`。

## Approaches(已定,含理由)

1. **session_id ↔ conversations 映射**:前端 uuid `session_id` 存进 `conversations.user_id VARCHAR(64)`,`messages.conversation_id` 引用自增 `id`。前端零改动。(用户已确认)
2. **SQLAlchemy 同步 + PyMySQL**:`sessionmaker` + 声明式模型。学习项目、库负载极小,同步最简;事件循环阻塞毫秒级可忽略。工具函数保持同步签名。
3. **运行时上下文:内存为主 + DB 只写落库**(用户已选):模型上下文仍用现有内存会话(`trim_history` 裁剪语义不变),消息按序写进 messages 表做记录。重启丢上下文但历史在库。
4. **单轮流程**:`llm.bind_tools(tools).ainvoke(messages)` 判定 → 若有 `tool_calls` 逐个执行并回灌 `ToolMessage` → 最终答案 `llm.astream(messages)` 逐 token 流式;最终收敛用**不带工具的 base llm**,机制上杜绝二次工具调用。
5. **工具内部开 DB 会话**:工具签名只暴露业务参数;query_faq/create_ticket 在函数体内经模块级 `SessionLocal` 开/关会话。`create_ticket` 的 `conversation_id` 用 `ContextVar` 由执行器注入,不进工具 Schema。
6. **工具出错**:捕获后把错误文案作为 `ToolMessage` 回灌给模型,由模型组织一句自然说明继续流式(不直接断流)。
7. **历史回放**:内存历史只存「user + 最终 assistant 答案」(工具调用/结果的消息是每轮内部的、不回放给模型);DB 里四种消息(user/assistant 工具调用/tool 结果/assistant 答案)全落。
8. **枚举可移植**:ORM 模型忠实映射 DDL;ENUM 列用 `Enum(..., native_enum=False)`(存 VARCHAR),使单测跑 SQLite 内存库、生产跑 MySQL。
9. **模型选工具可信度验证**:`qwen3:8b` + 5 选 1 + 可选不调是本章最不稳处;不套 TDD,改用**标注样例集**跑一遍验证选工具正确率(含「邮费→查不到」负例)。
10. **前端改造走 Vibe Coding**(用户指定):不套 brainstorm/TDD/code review,我描述效果、用户反馈迭代。

## How(已定细节)

- **DDL**:`db/schema.sql`(4 张表,即用户提供的 DDL 落盘)。种子数据灌 faq(含「退货政策」答案,不含「邮费」关键词,制造验收 #3 的预期漏召回)。
- **新依赖**:`sqlalchemy`、`pymysql`、`cryptography`(MySQL 8 认证)。
- **配置**:`.env` 加 `DATABASE_URL=mysql+pymysql://root:123456@127.0.0.1:3306/mew_help`;`Settings` 加 `database_url` 字段。
- **Docker**:`docker-compose.yml` 起 `mysql:8`,env `MYSQL_ROOT_PASSWORD=123456`、`MYSQL_DATABASE=mew_help`,端口 `3306:3306`,volume 持久化,挂载 `db/schema.sql` 初始化建表。
- **5 个工具**(`@tool`,Pydantic 参数成 Schema):
  - `query_order(order_id: str) -> str` — mock 随机订单状态。
  - `query_product(product_name: str) -> str` — mock 随机商品信息。
  - `query_logistics(order_id: str) -> str` — mock 随机物流状态。
  - `query_faq(keyword: str) -> str` — `LIKE %keyword%` 查 faq,无结果返回「未查到」。
  - `create_ticket(description: str, ticket_type: str) -> str` — 插 tickets,工单号 `T + YYYYMMDDHHMMSS + 3 位序号`,返回工单号。
- **工具基础设施**:注册表 `dict[name → tool]`;执行包装器做 try/except 错误捕获 + `asyncio.wait_for` 超时 + 失败重试;结果字符串化回灌。
- **SSE 事件**(`/chat` 输出):`meta` / `token` / **`tool`(新增:`{type:"tool", name, status:"running"|"done"|"error", message?}`)** / `done` / `error`。
- **落库**:`conversations`(首次建,`user_id`=session_id);`messages` 依次写:user、assistant 工具调用(`content` 空、`tool_calls` JSON)、tool 结果(`role="tool"`、`tool_call_id`)、assistant 最终答案。
- **前端**(Vibe Coding):`app.js` 处理 `tool` 事件 → 在客服气泡上方/内渲染「🔧 工具名」徽章(running 转圈 → done 打勾);其余流式不变。

## 测试 / 验收

- 自动化(pytest):DB 模型/工具注册/执行包装器(超时重试、错误处理)/落库逻辑,用 SQLite 内存库跑;既有 24 测试不回归。
- 标注样例集(替代 TDD,针对「模型选哪个工具」):约 8~10 条 `(用户话术 → 期望工具)`,含「订单 1001 物流→query_logistics」「退货政策→query_faq」「邮费是多少→查不到(负例)」「你好→不调工具」等,脚本跑一遍打正确率。
- 手动验收(3 条,见用户验收标准):浏览器问物流/退货政策/邮费。

## 留给实现者

- SQLAlchemy 模型字段与 DDL 的精确对应、种子数据具体内容。
- 工具包装器(超时/重试/错误)的精确参数(超时秒数、重试次数)。
- 单轮流程在 `ChatService.stream_chat` 里的具体重构(含 ContextVar 注入、消息落库顺序、内存历史只存 user+最终答案)。
- 标注样例集的具体条目与打分脚本。
- 前端工具徽章的样式与渲染逻辑(Vibe Coding)。
