# 第八章「工具系统」— 设计文档

日期:2026-11-21
状态:待 builder check / 待用户批准

## 1. 意图与动机

ch02 到 ch06 的工具是 6 个写死的 `@tool` 函数,硬编码在 `app/tools/__init__.py`(`ALL_TOOLS` → `TOOL_REGISTRY` dict);`AGENT_TOOLS` 在 import 时算好,`react_agent_node` 每轮 `bind_tools` 用这份静态表。要加一个工具得改这份清单、改 agent、重启服务;也没有校验、权限、审计、统一的重试/超时/错误分诊——`ToolExecutor.execute` 就 120s 超时 + 2 次无差别重试。

本章把工具层升级成「即插即用的工具系统」:一个统一注册中心(内置启动时登记 + MCP 连上动态发现,都带齐「工具名/用途描述/JSON Schema」三件套)、一个统一执行引擎(校验 → 权限 → 执行 → 分诊 → 格式化 → 审计)、`tool_audit_logs` 审计留痕、两个自建业务 MCP Server(物流/售后,Streamable HTTP)接入,以及 create_ticket 走 LangGraph `interrupt` 的前端预览确认流。新工具(内置或 MCP)注册/发现即用,不改核心代码、不重启客服服务。

## 2. 目标 / 非目标 / 反目标

**目标**
1. 工具注册中心:内置 + MCP 统一三件套(name/description/JSON Schema),同一份表;内置启动时登记,MCP 现问现拿;新工具进来主力 Agent 即用。
2. 参数校验:执行前按 JSON Schema 校验;类型/必填/取值越界拦下,把错误说明包成工具结果回灌模型(追问或重组织调用),不抛异常了事。
3. 权限:读/写两类,写由执行引擎把门。本系统写操作只有 `create_ticket`,真正执行要等前端确认(resume)放行,没确认的调用引擎直接拒。外部 MCP 用途声明不可信,读/写只认我方权限规则(默认只读 + 显式白名单)。
4. 执行引擎:所有工具调用走同一处,统一管超时、重试、错误三分类、结果格式化。重试只给网络暂态,业务空不重试,写默认不自动重试。
5. 审计:`tool_audit_logs` 每次调用落一条(含被权限拒/被校验拦),不挂外键、写审计失败不拦工具执行。
6. MCP:自建物流/售后两个业务 Server(独立进程、Streamable HTTP、mock 数据),客服系统作 Client 接入;工具与内置合成一份,Agent 一视同仁;`query_logistics` 内置下线防重名。
7. 建工单确认流:明确要求建单 → Agent 核对必填(缺则追问)→ 凑齐发起 `create_ticket` → 引擎不直接执行 → `interrupt` 推工单预览 → 前端确认/取消 → resume 放行或按权限拒绝落审计。

**非目标**
- Skill 机制(生态概念,不落代码)。
- 更多外部系统 / 仓储等(两个业务 MCP 足够撑本章)。

**反目标(跑通也算失败)**
- 加一个工具要改 agent/graph/executor 核心代码或重启客服服务。
- 写操作超时被自动重试(重复执行比失败更糟)。
- 外部 MCP 的用途声明或模型临场判断来决定能不能调写操作。
- 审计写入失败反过来拦断工具执行。
- 校验失败抛异常把整条流打断(应回灌模型)。

## 3. 约束

- 技术选型定死:MCP Server 用官方 Python SDK(`mcp`,Streamable HTTP);MCP Client 用 `langchain-mcp-adapters` 的 `MultiServerMCPClient`。不换。
- 现有聊天上游(意图路由/检索/退款子流程)不动;第 5 章投诉流程「前端按钮建单」那条路(`POST /api/tickets`)保持原样。
- 涉及 `mcp` / `langchain-mcp-adapters` / LangGraph `interrupt` 的 API,动手前 Context7 核实(本环境被墙则 WebSearch 兜底)。
- `tool_audit_logs` DDL 按用户给定逐字保留(见 §5.1)。

## 4. 方案取舍(approaches)

### 4.1 工具注册中心的抽象
- **A(选)独立 `RegisteredTool` 记录 + `ToolRegistry`**:把「工具名/描述/JSON Schema/读写标记/来源/实现」包成一条记录,内置启动时 `register()`,MCP 每轮 `discover()` 现问现拿。agent 每轮从 registry 取并集 `bind_tools`。统一三件套是需求硬要求,MCP 工具(名字用**裸名、不加 server 前缀**,来源靠 `mcp_server` 字段标)不包一层就没法填「来源/读写」这些字段。**选**。
- B:直接复用 langchain `@tool` 对象 + 旁挂 sidecar 字典。少一层抽象,但 MCP 工具没有「来源/读写/确认」挂靠点,校验要对两份数据(`args_schema` 与旁挂 JSON Schema)容易漂移。**弃**。

### 4.2 create_ticket 确认流的 interrupt 落点
- **A(选)专用 `confirm_ticket` 图节点**。`interrupt()` 的语义是「resume 时所在节点从头重跑」(见 `fetch_order` 注释)。若在 ReAct 循环里调 `interrupt`,resume 会把整段 ReAct 循环重跑一遍——重调 LLM、重发工具调用、非确定。所以 agent 节点只在循环里**把 create_ticket 记为 `pending_ticket` 后 break**,真正 `interrupt` 放在一个无副作用的 `confirm_ticket` 节点里(和 `fetch_order` 同款)。**选**。
- B:在 ReAct 循环内直接 `interrupt`。写起来少一个节点,但 resume 重跑循环,LLM 二次调用可能不再调 create_ticket,确认结果被悬空。**弃**。

### 4.3 MCP 读/写判定
- **A(选)默认只读 + 我方配置白名单**。内置工具显式标 read/write;MCP 工具一律默认只读,只有名字出现在 `MCP_WRITE_TOOLS` 白名单(默认空)才算写。引擎只看这条,不看 Server 声明、不让模型判断。**选**。
- B:信任 MCP 工具 schema 里的注解/元数据判写。需求明说「用途声明不可信」,直接违背。**弃**。

### 4.4 路由「明确要求建单」
- **A(选)新增「建工单」意图**,`_ROUTE_BY_INTENT["建工单"] = "business"` 进 Agent(Agent 有 create_ticket)。投诉安抚+按钮那条路不动。用户已拍板。**选**。
- B:投诉意图整体改路由进 Agent。动第 5 章确定性安抚行为,与「按钮路保持不动」摩擦。**弃**。

## 5. 已定设计(how)

### 5.1 数据模型(按用户给的 DDL)

`tool_audit_logs`(用户给定,逐字保留):`id` / `conversation_id`(BIGINT NULL,普通索引)/ `tool_call_id`(VARCHAR64 NULL)/ `tool_name` / `tool_source`(ENUM builtin/mcp)/ `mcp_server`(VARCHAR64 NULL)/ `arguments`(JSON)/ `result_summary`(TEXT)/ `status`(ENUM 成功/失败/超时/校验拦下/权限拒绝)/ `error_message`(VARCHAR512 NULL)/ `retry_count`(TINYINT UNSIGNED 默认 0)/ `duration_ms`(INT UNSIGNED NULL)/ `created_at`。不挂外键,`conversation_id` 只建普通索引。

完整 `CREATE TABLE` 按用户给定逐字进 `db/schema.sql`(含 `SET NAMES utf8mb4` 守卫、`idx_conversation_id`/`idx_tool_name`/`idx_status` 三个普通索引、ENGINE=InnoDB CHARSET=utf8mb4、无外键)。ORM `ToolAuditLog` 挂 `Base.metadata`,`migrate.py` 的 `create_all` 幂等补表(存量库),`docker-compose` 首次 init 走 `schema.sql`。

### 5.2 工具注册中心 `app/tools/registry.py`

```python
@dataclass
class RegisteredTool:
    name: str            # 工具名(唯一,agent 按此绑定)
    description: str     # 用途描述
    json_schema: dict    # JSON Schema 参数定义(校验 + 喂模型)
    read_only: bool      # True=读,False=写(写要确认/白名单)
    source: str          # "builtin" | "mcp"
    mcp_server: str | None
    impl: object         # 可 ainvoke 的 langchain BaseTool / 包装后的异步可调用
```

- 内置:启动(import)时 `registry.register(name=..., description=..., json_schema=..., read_only=..., impl=tool)`。加一个内置工具 = 写一个 `@tool` + 一行 `register`,不动 agent/graph/executor。
- MCP:`registry.discover_mcp(client)` 每轮对每个已配置 server 调 `client.get_tools(server_name=name)`,逐个包成 `RegisteredTool(source="mcp", mcp_server=name, read_only=默认 True 除非名在白名单)`。
- `registry.agent_tools()`:返回给 `bind_tools` 的并集 = 全部已注册工具 **去掉 `query_faq`**(检索由 knowledge 节点做,避免双检索);`query_logistics` 直接不注册(下线)。`create_ticket` **重新纳入**。
- **命名**:MCP 工具用裸名(无前缀),物流 Server 提供的 `query_logistics` 与内置同名、故内置不注册;审计 `tool_name` 记裸名、`mcp_server` 记来源 Server;`MCP_WRITE_TOOLS` 白名单填裸名。

### 5.3 执行引擎 `app/tools/executor.py`(重写)

统一入口 `async execute(call) -> ToolResult`,call 携带 `name / raw_args / tool_call_id / conversation_id`。顺序:

1. **查表**:registry 里找不到 → `ToolResult(kind="error", msg="未找到工具")`。
2. **校验**:用 `jsonschema` 对 `raw_args` 按 `json_schema` 校验(内置取 `impl.args_schema.model_json_schema()`,MCP 取 `impl.args_schema` 已是 dict,统一归一成 dict)。不合法 → `kind="validation_error"`,msg=逐条校验错误;审计 status=「校验拦下」;**不执行**。
3. **权限**:`read_only=False` 的 MCP 工具若不在 `MCP_WRITE_TOOLS` 白名单 → `kind="permission_denied"`,审计「权限拒绝」。内置 `create_ticket`(写)合法 → **不执行**,返回 `kind="needs_confirmation"` + `draft={ticket_type, description}`(预览数据)。
4. **执行(读)**:`asyncio.wait_for(impl.ainvoke(args), timeout)`;`read_only` 的重试只对「网络暂态」异常(超时 / MCP 传输/会话异常),最多 `TOOL_RETRIES` 次,业务空结果不重试;写操作重试次数恒 0。
5. **三分类**:参数不合法(校验,步骤 2)/ 查询落空(工具返回空、`refused=True`、空串/「未查到」这类正常返回)/ 真故障(超时、网络、DB、未捕获异常)。坏消息如实回模型。**重试只认「网络暂态」异常(超时/传输/会话),正常返回(查到或查空)一律不重试**;两个 MCP 业务工具返回普通字符串(照第 2 章 mock),「查询落空」即正常返回、不重试。
6. **格式化**:只挑回答用得到的字段;内部枚举码翻人话;MCP `ainvoke` 返回 `(content, artifact)` 元组时解包取 content、块列表转字符串;JSON 序列化 `ensure_ascii=False`。
7. **审计**:每次调用落一条(成功/失败/超时/校验拦下/权限拒绝),写 `retry_count / duration_ms / arguments / result_summary(截断) / error_message`;审计写入 `try/except` 隔离,失败只 log 不抛。

`ToolResult`:`kind ∈ {ok, validation_error, permission_denied, needs_confirmation, error}`,`content`(给模型的文本),`draft`(needs_confirmation 时)。

### 5.4 权限模型

- 内置:注册时显式 `read_only`。`create_ticket`=`read_only=False`(写),其余=`True`。
- MCP:默认 `read_only=True`;只有名在 `settings.mcp_write_tools`(list,默认空)的才算写。引擎对 MCP 写工具一律要求白名单命中,否则「权限拒绝」;本章白名单空,故两个业务 MCP 全只读。
- `submit_refund` 保持 ch06 的 agent 循环内拦截(合成 `refund_form` 动作 + ToolMessage),不进引擎、不审计——它是「交前端表单」的伪工具,不是真执行。这是刻意的例外,记于代码注释。

### 5.5 MCP Server(两个,独立进程)

`mcp_servers/logistics_server.py` + `mcp_servers/aftersale_server.py`,用官方 SDK v2 的 `MCPServer`(v1 的 `FastMCP` 已改名,旧 import 路径删除):

```python
from mcp.server import MCPServer
mcp = MCPServer("logistics")

@mcp.tool()
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流/包裹运输状态。"""
    ...  # 随机 mock,不接真实系统、不建表

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8100)
```

- 物流:一个工具 `query_logistics`(随机状态,仿第 2 章)。
- 售后:两个工具 `check_warranty`(查在保)+ `check_return_progress`(查退货进度)。
- 端口:`logistics=8100`、`aftersale=8101`(可配置),默认路径 `/mcp`。名字/描述/参数 schema 由函数名/docstring/type hints 派生。
- 返回**普通字符串**(照第 2 章 mock),无 JSON 信封;「查询落空」由引擎当正常返回处理、不重试。

### 5.6 MCP Client `app/mcp/client.py`

- `MultiServerMCPClient({"logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url}, "aftersale": {...}})`(注意:旧 `async with MultiServerMCPClient(...)` 已删除,现用 `await client.get_tools(...)` / `client.session(name)`)。
- `discover()`:对每个 server `await client.get_tools(server_name=name)`,把返回的 langchain 工具包成 `RegisteredTool`(source=mcp、mcp_server=name、json_schema 从 `tool.args_schema` 归一)。
- 现问现拿:每轮 agent 前 `discover()` 一次,满足「只重启 Server 就能用上新工具」。
- **生命周期/并发**:内置 registry 启动时写一次、之后只读;`MultiServerMCPClient` 惰性建单例(只持连接配置,不含会话态);`discover()` 每轮**返回一份新的本地工具表**(不改共享全局),agent 节点把内置 + MCP 并集后 `bind_tools`,天然无跨会话写竞争。
- **Server 宕机**:`discover()` 对连不上的 server 记 warn 后**跳过该 server 的工具**(降级,不崩整轮;不缓存旧工具);执行时 Server 宕机由引擎超时/重试/分诊兜底,如实回模型。

### 5.7 create_ticket 确认流(图)

- `GraphState` 加 `pending_ticket: dict | None`(reset 清空)。
- `react_agent_node`:
  - `tools = registry.agent_tools()`,`bound_llm.bind_tools(tools)`(每轮动态取)。
  - 循环里对每个 tool_call:`submit_refund` 走 ch06 拦截;其余走 `exec_.execute(call)`。
  - 若 `result.kind == "needs_confirmation"` → 记 `pending_ticket = result.draft` + 该 `tool_call_id`,**break**(不再往下执行);否则 ToolMessage(content=result.content)。
  - 校验拦下/权限拒绝/错误都作为 ToolMessage 回灌,循环继续(模型追问或重组织)。
- `create_ticket` 的 JSON Schema 强制 `description` 非空(minLength ≥ 1)且 `ticket_type ∈ {售后,投诉,咨询}`(退款走 refund 表单不走建单工具),故「缺必填/类型非法」在确认之前就拦下回灌,不会走到「确认后 DB 插不进」。`pending_ticket` 存**完整已校验参数 `args`**(预览由 args 派生),将来加字段不丢。
- 新增 `confirm_ticket` 节点:
  - `decision = interrupt({"type": "ticket_preview", "draft": pending_ticket})`;resume 值 `{"confirm": bool}`。
  - `confirm=True` → 引擎写路径执行 create_ticket(落 `tickets`,ContextVar 设 conversation_id),审计「成功」;`confirm=False` → 不执行,审计「权限拒绝」。
  - 返回最终 `AIMessage`:确认 → `工单已创建,工单号 {ticket_no}…`;取消 → `已取消建工单…`;清 `pending_ticket`。
- 边:`agent` →(pending_ticket 有值)→ `confirm_ticket` → `log`;否则 `agent` → `log`。`confirm_ticket` 不回头进 agent(最终回复固定话术,确定性、可测)。

### 5.8 路由与意图

- `nodes.INTENTS` / `VALID_INTENTS` 加「建工单」;`_ROUTE_BY_INTENT["建工单"] = "business"`(进 Agent)。
- `INTENT_SYSTEM_PROMPT` 枚举加「建工单」+ 边界 few-shot(明确说「帮我建个工单/我要投诉并建单」→ 建工单)。

### 5.9 chat_service 与 resume 契约

- `_stream_invoke` 的 interrupt 帧从硬编码 `orders` 泛化为:按 `payload["type"]` 分支——`select_order` → `orders`,`ticket_preview` → `draft`;帧里带 `kind`。
- `stream_resume(conversation_id, payload)`:从只收 `order_id` 字符串泛化为透传 `payload`(order 选择=字符串 `order_id`,工单确认=`{"confirm": bool}`),`Command(resume=payload)`。
- 路由:`/api/actions/resume` 的 `ResumeRequest` 扩展为 `conversation_id + (order_id | confirm)`(或等价判别式),order 选择路径向后兼容。

### 5.10 配置 `app/core/config.py`

新增:`mcp_logistics_url`、`mcp_aftersale_url`、`mcp_write_tools: list[str] = []`、`tool_timeout_seconds: float = 120.0`、`tool_retries: int = 2`。`.env.example` 同步补这几个字段(有默认值)。

### 5.11 前端(工单预览卡片,Vibe Coding)

- `dispatch` 遇 `interrupt` 且 `kind == "ticket_preview"` → 渲染工单预览卡片(工单类型 + 问题描述 + 「确认提交」「取消」两按钮)。
- 确认 → `POST /api/actions/resume {conversation_id, confirm: true}`;取消 → `{confirm: false}`。复用现有 `streamTo` 渲染续流结果。
- 订单选择卡片路径(`kind == "select_order"`)不动。

## 6. 留给 builder 的细节

- `jsonschema` 版本与 `mcp` / `langchain-mcp-adapters` 与 `langchain-core 1.8` 的兼容版本号,在实现前用 Context7/WebSearch 核对再钉进 `requirements.txt`。
- MCP 工具 `ainvoke` 返回 `(content, artifact)` 及 content 块的具体结构(字符串 / `ToolMessage` / content block 列表)的确切归一化。
- `client.get_tools(server_name=...)` 返回的 langchain 工具 `args_schema` 是 pydantic 还是 dict,据此归一成 JSON Schema。
- `tool_audit_logs.result_summary` 的截断长度与「只挑回答用得到的字段」的字段白名单。
- `confirm_ticket` 节点里工单号生成与 `tickets` 写入的复用(与 `db_tools.create_ticket` 合一,避免两处生成工单号)。
- 两个 MCP Server 的启动脚本/命令(供验收 2「单独起进程」)。

## 7. 验收映射

1. 注册即用:新增内置工具 = `@tool` + `registry.register(...)` 一行,agent 每轮动态取 → 不改核心代码。
2. 两个 Server 单独起进程,物流问题经 MCP 查到数据 → 内置 `query_logistics` 下线,MCP 提供同名工具,Agent 走引擎调 MCP。
3. Server 侧加工具、只重启该 Server → `discover()` 每轮现问现拿,客服侧不重启。
4. 说「帮我建个工单」没讲清 → 校验拦下回灌 → Agent 追问;凑齐 → `needs_confirmation` → `confirm_ticket` interrupt → 前端预览卡 → 确认 → `tickets` 落一条 + 回复带工单号。
5. 预览点取消 → 不建单,`tool_audit_logs` 该条 `status=权限拒绝`。
6. 人为超时 → 读工具重试 + 兜底,审计重试次数/超时/耗时齐全;写操作超时 → 不自动重试(`retry_count=0`)。
