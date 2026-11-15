# 电商智能客服系统 — 第一章「纯对话」设计文档

日期:2026-11-15
状态:builder 评审已过,待用户批准

## 1. 意图与动机

这是电商智能客服系统的**地基章节**:先跑通纯对话,验证技术栈(直连上游 OpenAI 协议 + 模型可换)能否支撑后续功能,用最小闭环落地。后续章节会加工具调用、Agent 循环、RAG、会话持久化,因此本章的结构要为这些留扩展位,但本章**不实现**它们。

## 2. 目标 / 非目标 / 反目标

### 目标
1. **对话接口 `/chat`**:多轮对话,SSE 流式输出、逐 token 推送。
2. **Prompt 管理**:`PromptTemplate` 模板化,System Prompt 写清客服角色 + 行为约束。
3. **结构化输出 `/extract`**:把售后描述提取成 `{订单号, 诉求类型, 期望方案}`,用 `with_structured_output` 实现。
4. **多轮上下文(最简版)**:历史消息裁剪 + token 预算控制。

### 非目标(本章不做)
- 工具调用、Agent 循环。
- 会话持久化(数据库)。
- 用户认证、多租户、限流。
- 聊天 Web 页面(后续用 Vibe Coding 单独做,不走本章流程)。

### 反目标(即使"能跑"也算失败)
- 不能为"模型可换"引入复杂 provider 抽象层,导致后续扩展反而更难。
- 不能让结构化输出在某家模型上**悄悄失败**:必须在文档核实 + 测试覆盖,不支持时明确报错/降级,接口不变。
- 不能把历史裁剪做复杂:保持最简,后续章节再演进。

## 3. 约束(定死,不可自行改)

- 技术栈:Python + FastAPI + LangChain。
- 模型接入:**应用侧统一 OpenAI 协议**,地址、模型名、密钥全在 `.env`;GPT / Claude / DeepSeek / Ollama 都能换。
- 涉及库/框架 API 用法,先用 Context7 MCP(不可用时用 WebFetch 抓官方文档)查最新接口再动手。
- 若实现中发现技术选型矛盾或走不通,**停下来问,不自行换方案**。

## 4. 方案(Approaches)

### 4.1 LLM 客户端:单一 `ChatOpenAI` + 可配 base_url(选定)

应用侧统一说 OpenAI 协议,用 `langchain-openai` 的 `ChatOpenAI`,从 `.env` 读 `base_url / model / api_key`:

| Provider | base_url | 说明 |
|---|---|---|
| OpenAI (GPT) | `https://api.openai.com/v1` | 默认 |
| Claude | `https://api.anthropic.com/v1` | Anthropic 的 OpenAI 兼容端点 |
| DeepSeek | `https://api.deepseek.com/v1` | 官方 OpenAI 兼容 |
| Ollama | `http://localhost:11434/v1` | 本地 OpenAI 兼容 |

**为什么不用 provider 抽象层**(每家单独 SDK):用户明确"应用侧统一 OpenAI 协议",四家都有 OpenAI 兼容端点;抽象层是过度设计,本章用单一客户端,后续真有差异再按需加一层工厂。

### 4.2 结构化输出:`with_structured_output(PydanticModel)`(选定)

`langchain` 的 `with_structured_output`,底层走 **function calling(tools)**。这是唯一的技术风险点:各家对 JSON Schema `strict`、tool use 的支持程度不一。

**降级方案(接口不变)**:某模型不支持 function calling 时,改用 **JSON mode + Pydantic 校验**,再兜底 **prompt 约束 + 正则提取**。实现时用文档逐家核实,真走不通停下来问。

### 4.3 流式:FastAPI `StreamingResponse` + SSE(选定)

`/chat` 用 `StreamingResponse(media_type="text/event-stream")`,逐 token `yield`。不用 `sse-starlette`,标准库手写 SSE 即可,少一个依赖。

### 4.4 考虑过但放弃
- **每 token 一段完整消息**(非 SSE,直接返回):不满足"逐 token 推送"的验收。
- **LangServe / langserve**:引入额外框架,本章用不到,徒增复杂度。
- **langchain 的 ChatPromptTemplate 从 YAML 文件加载**:可用,但本章先内联在 `prompts/` 的 Python 模块里用 `PromptTemplate`/`ChatPromptTemplate` 构造,后续要外部化再迁移,避免过早引入模板文件加载机制。

## 5. 已定 / 默认的 How

### 5.1 项目结构(我的默认)
```
Mew_help/
  app/
    main.py                    # FastAPI 实例 + 挂路由
    api/routes/
      chat.py                  # POST /chat  (SSE 流式)
      extract.py               # POST /extract (结构化 JSON)
    schemas/
      chat.py                  # ChatRequest / ChatMessage / SSE 事件模型
      extraction.py            # AfterSalesExtraction (with_structured_output 的 schema)
    services/
      chat_service.py          # 流式对话编排
      extraction_service.py    # 结构化提取编排
      history.py               # 历史裁剪 + token 预算
    prompts/
      system.py                # 客服 System Prompt(角色 + 行为约束)
    core/
      config.py                # pydantic-settings 读 .env
      llm.py                   # ChatOpenAI 工厂(读 config 拼 base_url/model/key)
  tests/                       # pytest,LLM 用 mock,不依赖真实模型
  .env.example
  pyproject.toml
  dev-notes/ch01.md
  docs/superpowers/specs/
```

### 5.2 结构化输出 Schema(已定,6 类枚举)
```python
class AfterSalesExtraction(BaseModel):
    order_id: str | None            # 订单号,描述没提就 None
    request_type: Literal["退款","退货","换货","维修","物流查询","其他"]
    expected_resolution: str        # 期望方案,自由文本(顾客想要的解决方式)
```
- `request_type` 用枚举(6 类),可约束模型;`expected_resolution` 用自由文本(方案天然开放)。
- 来源:我提议的默认,用户未反对(审批时可改)。

### 5.3 Ollama 默认(已定:Ollama 本地)
- `base_url=http://localhost:11434/v1`,默认模型 **`qwen2.5:7b`**(中文好、支持 function calling)。
- **结构化输出建议拉 `qwen2.5:14b`**:7b 的 function calling 偏弱,是结构化输出的最大风险点;14b 更稳。7b 也能跑,但更依赖 4.2 的 JSON-mode 降级链。
- 用户尚未安装 Ollama:代码与测试不依赖真实模型(测试用 mock);演示前需装好并 `ollama pull` 对应模型。

### 5.4 多轮上下文与历史裁剪(我的默认)
- 保留 System Prompt + 最近 **N=10** 条历史消息。
- token 预算:历史部分 ~**4000** token,超预算从最旧开始裁剪。
- token 计数:优先 `tiktoken`(`cl100k_base`),若 Python 3.14 下无 wheel 则退化为 `len(text)/4` 粗估。实现时定。

### 5.5 多轮会话契约(服务端内存会话,我的默认)
- `/chat` 采用 **session_id + 服务端内存会话**:首次请求不带 session_id,服务端新建会话并在 SSE 首事件里返回 `{"type":"meta","session_id":"..."}`;后续请求带 session_id,服务端从内存 dict 取历史,做裁剪 + token 预算。
- 内存 dict,无 DB、无认证;重启即失,本章可接受。
- 来源:我推荐的方案(用户未反对);备选是无状态客户端带历史,已否决——curl 验收第 2 条要手动拼历史、状态压在客户端,且不符后面 Web 前端的形态。

### 5.6 SSE 事件格式(我的默认)
```
data: {"type":"meta","session_id":"abc123"}\n\n
data: {"type":"token","content":"你"}\n\n
data: {"type":"token","content":"好"}\n\n
...
data: {"type":"done"}\n\n
```
错误时:`data: {"type":"error","message":"..."}\n\n`。

### 5.7 System Prompt 内容(我的默认:通用占位)
- 先写一个通用电商客服角色:System Prompt 用 `ChatPromptTemplate` 组装,含角色定位(某电商平台售后客服)、行为约束(礼貌、专业、不臆造订单信息、无法处理时引导转人工)、输出约束。
- 品牌名/退换货政策/话术为**占位符**,后续可注入;本章不要求真实品牌。

### 5.8 Python 版本(我的默认:先验证 3.14,卡则切 3.12)
- 当前环境 Python 3.14.7,较新,`pydantic-core`/`langchain`/`tiktoken` 可能缺 wheel。
- 装依赖列为实现第一步:若 3.14 装不上,切 Python 3.12(或 3.11)建 venv,不因此改技术栈。

## 6. 留给实现者

- 具体函数签名、路由参数、错误码约定。
- token 计数的最终实现(tiktoken vs 粗估)。
- 结构化输出降级链(JSON mode / prompt 兜底)的具体触发条件。
- 测试用例设计(TDD:先用 mock LLM 写测试)。

## 7. 验收标准(来自用户,不可改)

1. `curl` 调 `/chat` 能看到流式回复。
2. 连续问两轮,第二轮能接住第一轮上下文。
3. 发一段售后描述给 `/extract`,能拿到结构化 JSON。
