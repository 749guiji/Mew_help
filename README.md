# Mew Help — 电商智能客服系统

一个本地可跑的电商售后智能客服 RAG 系统，从纯对话、Function Calling 工具链、RAG 知识库、混合检索 + 重排 + 评估，一路做到 LangGraph 生产级编排、分流器正式版、三层会话上下文管理，以及即插即用的工具系统 + MCP 接入。

## 功能

- **ch01 纯对话 + 聊天前端**：FastAPI + LangChain，`/chat` SSE 流式输出 + `/extract` 结构化信息抽取；`static/` Mew 猫系界面，逐字渲染 SSE、多轮对话、工具调用徽章。
- **ch02 Function Calling 工具链**：5 个工具（订单/商品/物流 mock、FAQ 查询、创建工单），单轮工具链 + 消息落库 MySQL（`conversations`/`messages`/`tickets`/`faq`）。
- **ch03 RAG 知识库**：`knowledge_chunks` 权威源 + Milvus Lite 向量化双写幂等 + 离线 QA 抽取暂存；`/kb` 后台 + `/api/kb` 接口。
- **ch04 混合检索 + 重排 + 评估**：
  - Milvus Lite 内置 **BM25**（jieba 中文分词）与 **BGE-M3** dense 向量，`hybrid_search` + **RRF** 融合。
  - **bge-reranker-v2-m3** 精排（Cross-Encoder），Lost-in-the-Middle 首尾重排。
  - Query 改写 + 同义词检索侧扩写、元数据过滤。
  - 生成质量控制：引用编号溯源、自评不足拒答入池、忠实度校验。
  - 四策略评估（纯向量 / 纯 BM25 / 混合 RRF / 混合 + 精排）：Recall@K、MRR、证据覆盖度、答案覆盖度、Faithfulness、拒答率。
- **ch05 LangGraph 生产级编排**：`StateGraph` 确定性编排骨架 + 手写 **ReAct 主力 Agent**；七类意图分流；指代消解 / 意图识别；`InMemorySaver` checkpointer 多轮持久化；转人工 / 建工单双按钮前端。
- **ch06 分流器正式版**：指代消解 + Query 改写（结合历史补全成不依赖上下文的完整问题）；Query 扩写（强制 JSON `queries[]`）；意图识别 JSON 四件套（七类枚举 + 强制 JSON `{intent,confidence}` + 边界 few-shot +「其他」兜底）；**5 出口**分流；`interrupt()`/`resume` 挂起恢复（订单选择器卡片 + 退款表单）；退款单复用 `tickets`（`ticket_type='退款'`）。
- **ch07 会话上下文管理**：三层上下文（层1原文 / 层2截短 / 层3异步摘要）；两锚点 `summary_upto_msg_id` + `layer1_from_msg_id`，降级只挪 id 不搬数据；`conversation_summaries` 分段追加摘要；预算从窗口倒推 + 启动自检；`user_id`/`conversation_id` 解耦 + 前端多会话侧栏。
- **ch08 工具系统**：统一注册中心 `RegisteredTool` + `ToolRegistry`（内置启动登记、MCP 现问现拿，三件套 name/描述/JSON Schema）；统一执行引擎（JSON Schema 校验→错误回灌、读/写权限把关、超时/重试三分类、结果裁剪人话化、`tool_audit_logs` 审计留痕）；两个自建 MCP Server（物流/售后，Streamable HTTP）+ `MultiServerMCPClient` 接入；`create_ticket` 走 LangGraph `interrupt` 预览确认流（前端确认/取消）。

## 技术栈

FastAPI · LangChain · LangGraph · Ollama（`qwen3:8b` + `bge-m3` + `bge-reranker-v2-m3`）· MySQL 8（Docker）· Milvus Lite · SQLAlchemy · PyMilvus · sentence-transformers / torch · MCP（`mcp` SDK + `langchain-mcp-adapters`）

## 目录结构

```
app/            # FastAPI 应用（api/core/db/graph/jobs/kb/services/tools/prompts/schemas）
  graph/        #   LangGraph 图编排：build/state/router/nodes/agent/context(三层拼装)/budget/degradation/summarizer
  mcp/          #   MCP Client（discover_mcp_tools 现问现拿）
  tools/        #   工具层：registry(注册中心)/executor(执行引擎)/audit(审计)
mcp_servers/    # 自建业务 MCP Server（物流/售后,Streamable HTTP,独立进程）
knowledge/      # 知识库 Markdown 源（售后手册/商品目录/FAQ/退货政策）
evaluation/     # 评估集 + 检索/生成段评估脚本
scripts/        # 独立脚本（demo_agent_loop / eval_intent / eval_coref / eval_expand / eval_summary …）
static/         # 聊天 /kb /admin 三页前端
db/             # MySQL schema.sql
dev-notes/      # 各章节开发记录（ch01–ch08）
tests/          # pytest 测试
```

## 快速开始

### 1. 前置

- Python 3.14 + 依赖：`pip install -r requirements.txt`
- [Ollama](https://ollama.com) 已拉取模型：
  ```bash
  ollama pull qwen3:8b
  ollama pull bge-m3
  ```
  重排模型 `bge-reranker-v2-m3` 由 sentence-transformers 在本地 HF 缓存加载（首次运行时需可联网下载）。

### 2. 配置

```bash
cp .env.example .env   # 按需修改 MODEL / DATABASE_URL 等
```

> ch07 上下文预算相关字段（`MODEL_CONTEXT_WINDOW` / `DESIRED_ROUNDS` / `MAX_AGENT_STEPS` 等）均有默认值；ch08 的 MCP 地址（`MCP_LOGISTICS_URL` / `MCP_AFTERSALE_URL`）与工具超时/重试（`TOOL_TIMEOUT_SECONDS` / `TOOL_RETRIES`）也均有默认值。见 `app/core/config.py`，通常无需改动。

### 3. 起 MySQL + 灌知识库

```bash
docker compose up -d    # 启动 MySQL，首次会自动执行 db/schema.sql
# 存量库补建后续章节新增的表（幂等）
PYTHONPATH=. .venv/Scripts/python.exe -m app.jobs.migrate
# 灌入 FAQ 种子（MySQL 中文需 utf8mb4，已内置处理）
PYTHONPATH=. .venv/Scripts/python.exe -c "from app.db import SessionLocal; from app.db.seed import seed_faq; seed_faq(SessionLocal())"
# 切分 knowledge/*.md 并向量化入库（Milvus Lite）
PYTHONPATH=. .venv/Scripts/python.exe -m app.jobs.ingest_docs
```

### 4. 起两个 MCP Server + 启动服务

```bash
# 两个业务 MCP Server 各自独立进程(物流/售后,Streamable HTTP)
.venv/Scripts/python.exe -m mcp_servers.logistics_server   # 127.0.0.1:8100
.venv/Scripts/python.exe -m mcp_servers.aftersale_server   # 127.0.0.1:8101

# 客服服务
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 或 make dev
```

浏览器打开 <http://127.0.0.1:8000/>（聊天页）、<http://127.0.0.1:8000/kb>（知识库后台）、<http://127.0.0.1:8000/admin>（总览）。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 聊天 SSE 流式（支持 `user_id` / `conversation_id` / `order_id`） |
| POST | `/extract` | 结构化信息抽取 |
| GET | `/api/conversations?user_id=` | 列某用户会话 |
| GET | `/api/conversations/{id}/messages` | 回载会话历史 |
| GET | `/api/orders?user_id=` | 按用户列订单（订单选择器数据源） |
| POST | `/api/tickets` | 建工单（转人工 / 退款单复用此表） |
| POST | `/api/actions/resume` | 挂起子流程续跑（订单选择 `order_id` / 建工单确认 `confirm`） |
| POST | `/api/actions/create-refund` | 提交退款 |
| GET | `/kb` · `/admin` | 知识库 / 总览后台页 |
| GET/POST | `/api/kb/*` · `/api/admin/overview` | 知识库与后台接口 |
| GET/POST | `/api/jobs/{name}/status·log·start` | 后台任务状态 / 日志 / 触发 |

## 评估

```bash
# 四策略检索对比（R@K / MRR / 证据覆盖度）
python -m evaluation.run_eval --strategies dense,sparse,hybrid,hybrid_rerank

# 生成段：答案覆盖度（LLM-as-Judge，可加 --limit 子集）
python -m evaluation.run_eval --answer-coverage --limit 10

# 生成段：拒答率（知识库外问题）
python -m evaluation.run_eval --refusal

# 生成段：忠实度
python -m evaluation.run_eval --faithfulness --limit 20
```

> 注：`hybrid_rerank`/`--refusal` 会加载 torch + bge-reranker，与 Ollama 同时运行在 ~16GB 机器上可能内存吃紧，可用 `--limit` 子集或 `--strategies dense` 绕过重排。

意图 / 指代 / 扩写 / 摘要的专项评估脚本位于 `scripts/`（`eval_intent.py` / `eval_coref.py` / `eval_expand.py` / `eval_summary.py`）。

## 测试

```bash
pytest
```
