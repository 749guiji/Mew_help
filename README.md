# Mew Help — 电商智能客服系统

一个本地可跑的电商售后智能客服 RAG 系统。从纯对话、Function Calling 工具链、RAG 知识库、混合检索 + 重排 + 评估，一路做到 LangGraph 生产级编排、分流器正式版、三层会话上下文管理、即插即用工具系统 + MCP 接入、可观测性 + 数据飞轮，最后训练自己的**多标签主题分类器**给知识缺口批量归类。

全程留痕于 `dev-notes/`（ch01–ch10 逐章开发记录），`docs/superpowers/` 存每章的设计文档与实现计划。

## 功能

- **ch01 纯对话 + 聊天前端**：FastAPI + LangChain，`/chat` SSE 流式输出 + `/extract` 结构化信息抽取；`static/` 前端（翡翠绿 + 活力橙），逐字渲染 SSE、多轮对话、工具调用徽章。
- **ch02 Function Calling 工具链**：5 个工具（订单 / 商品 / 物流 mock、FAQ 查询、创建工单），单轮工具链 + 消息落库 MySQL（`conversations` / `messages` / `tickets` / `faq`）。
- **ch03 RAG 知识库**：`knowledge_chunks` 权威源 + Milvus Lite 向量化双写幂等 + 离线 QA 抽取暂存；`/kb` 后台 + `/api/kb` 接口。
- **ch04 混合检索 + 重排 + 评估**：
  - Milvus Lite 内置 **BM25**（jieba 中文分词）与 **BGE-M3** dense 向量，`hybrid_search` + **RRF** 融合。
  - **bge-reranker-v2-m3** 精排（Cross-Encoder），Lost-in-the-Middle 首尾重排。
  - Query 改写 + 同义词检索侧扩写、元数据过滤。
  - 生成质量控制：引用编号溯源、自评不足拒答入池、忠实度校验。
  - 四策略评估（纯向量 / 纯 BM25 / 混合 RRF / 混合 + 精排）：Recall@K、MRR、证据覆盖度、答案覆盖度、Faithfulness、拒答率。
- **ch05 LangGraph 生产级编排**：`StateGraph` 确定性编排骨架 + 手写 **ReAct 主力 Agent**；意图分流；指代消解 / 意图识别；`InMemorySaver` checkpointer 多轮持久化；转人工 / 建工单双按钮前端。
- **ch06 分流器正式版**：指代消解 + Query 改写（结合历史补全成不依赖上下文的完整问题）；Query 扩写（强制 JSON `queries[]`）；意图识别 JSON 四件套（多类枚举 + 强制 JSON `{intent,confidence}` + 边界 few-shot +「其他」兜底）；**多出口**分流；`interrupt()`/`resume` 挂起恢复（订单选择器卡片 + 退款表单）；退款单复用 `tickets`（`ticket_type='退款'`）。
- **ch07 会话上下文管理**：三层上下文（层1原文 / 层2截短 / 层3异步摘要）；两锚点 `summary_upto_msg_id` + `layer1_from_msg_id`，降级只挪 id 不搬数据；`conversation_summaries` 分段追加摘要；预算从窗口倒推 + 启动自检；`user_id`/`conversation_id` 解耦 + 前端多会话侧栏。
- **ch08 工具系统 + MCP**：统一注册中心 `RegisteredTool` + `ToolRegistry`（内置启动登记、MCP 现问现拿，三件套 name/描述/JSON Schema）；统一执行引擎（JSON Schema 校验→错误回灌、读/写权限把关、超时/重试三分类、结果裁剪人话化、`tool_audit_logs` 审计留痕）；两个自建 MCP Server（物流/售后，Streamable HTTP）+ `MultiServerMCPClient` 接入；`create_ticket` 走 LangGraph `interrupt` 预览确认流。
- **ch09 可观测性 + 数据飞轮**：
  - **Langfuse** 开源自部署（`langfuse/docker-compose.yml`），Graph 编译时挂一次 `CallbackHandler` 拿完整 trace 树；意图进 trace 元数据；未配 secret 静默跳过。
  - **evidence_confidence 三信号闸**替代简单得分阈值（Top1 分 / 分差 / 有效条数），低置信度落池并快照当轮召回片段。
  - **数据飞轮三入口**：检索置信度低 / 生成自评不足 / 用户点 👎，统一入 `low_confidence_questions`（标 `source` + 召回快照）。
  - **飞轮流水线**：标准化→查重→入 `review_queue` 待审（命中累加 `occurrence_count` 不新建行）；人工审核通过写回知识库（`content_type=faq` + 向量化）。
  - **评估流水线**：每轮落 `eval_runs`，按时间连成趋势。
- **ch10 多标签主题分类器**：把飞轮攒下的低置信度问题按主题批量归类，好决定先补哪块知识。
  - **17 类术语表**（`ml/taxonomy.py` 单一权威源）：退换货 / 物流 / 尺码 / 发票 / 质量问题 / 运费 / 优惠活动 / 价保 / 支付 / 订单修改 / 库存补货 / 商品信息 / 保修维修 / 账号 / 会员积分 / 评价 / 其他，含三条近邻硬规则（「修」归保修维修、「退」归退换货；运费管钱、物流管货；价保补差价、优惠券满减）。
  - **数据四步**：清洗 → 归并术语表 → 分层抽样（80-10-10）→ 增强只扩训练集；大模型预标 + 人工抽审。
  - **训练**：RoBERTa-wwm-ext 全参微调（多标签 sigmoid + BCE 头，早停 + checkpoint 断点续训，CPU 可跑）。
  - **评测**：每类 P/R/F1（macro 调和平均）+ 混淆矩阵 → `ml/artifacts/eval_report.json`。
  - **部署**：ONNX + 独立 FastAPI 推理服务（端口 8001），旁路批量归类写 `topic_classifications`，主链路不调。

## 技术栈

FastAPI · LangChain · LangGraph · Ollama（`qwen3:8b` + `bge-m3` + `bge-reranker-v2-m3`）· MySQL 8（Docker）· Milvus Lite · SQLAlchemy · PyMilvus · sentence-transformers / torch · MCP（`mcp` SDK + `langchain-mcp-adapters`）· Langfuse（自部署）· transformers · onnx / onnxruntime · ModelScope（模型下载）

## 目录结构

```
app/            # FastAPI 应用
  api/routes/   #   chat/extract/conversations/orders/tickets/actions/kb/admin/
                #   jobs/feedback/review/topics/acceptance 等路由
  core/         #   config/llm/budget/summarizer/degradation/jobs(作业运行器)
  db/           #   SQLAlchemy ORM 模型 + seed
  graph/        #   LangGraph 图编排：build/state/nodes/router/agent/
                #   context(三层拼装)/budget/degradation/summarizer/confidence
  jobs/         #   进程内作业：ingest_docs/qa_mining/flywheel/eval_run/classify_topics/migrate
  kb/           #   知识库：chunker/embedding/hybrid/milvus_store/retrieval/reranker/…
  mcp/          #   MCP Client（discover_mcp_tools 现问现拿）
  tools/        #   工具层：registry(注册中心)/executor(执行引擎)/audit(审计)
  services/     #   chat_service(图编排)/extraction/history/generation_quality
ml/             # ch10 主题分类器：taxonomy(术语表)/data(管线)/train/eval/
                #   export_onnx/predict/download/run_pipeline
inference/      # 独立 ONNX 推理服务（端口 8001）
mcp_servers/    # 自建业务 MCP Server（物流/售后，Streamable HTTP，独立进程）
langfuse/       # Langfuse 自部署 docker-compose
knowledge/      # 知识库 Markdown 源（售后手册/商品目录/FAQ/退货政策）+ 对话样本
evaluation/     # 评估集 + 检索/生成段评估脚本
scripts/        # 独立脚本（demo_agent_loop/eval_intent/eval_coref/eval_expand/eval_summary…）
static/         # 前端：聊天 /kb /admin /review /topics /acceptance 多页
db/             # MySQL schema.sql
dev-notes/      # 各章节开发记录（ch01–ch10）
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

> ch07 上下文预算字段（`MODEL_CONTEXT_WINDOW` / `DESIRED_ROUNDS` / `MAX_AGENT_STEPS` 等）、ch08 工具与 MCP 字段（`MCP_LOGISTICS_URL` / `MCP_AFTERSALE_URL` / `TOOL_TIMEOUT_SECONDS` / `TOOL_RETRIES`）、ch09 Langfuse 字段（`LANGFUSE_*`）均有默认值，见 `app/core/config.py`，通常无需改动。

### 3. 起 MySQL + 灌知识库

```bash
docker compose up -d    # 启动 MySQL，首次自动执行 db/schema.sql
# 存量库补建后续章节新增的表（幂等）
PYTHONPATH=. .venv/Scripts/python.exe -m app.jobs.migrate
# 灌入 FAQ 种子（MySQL 中文需 utf8mb4，已内置处理）
PYTHONPATH=. .venv/Scripts/python.exe -c "from app.db import SessionLocal; from app.db.seed import seed_faq; seed_faq(SessionLocal())"
# 切分 knowledge/*.md 并向量化入库（Milvus Lite）
PYTHONPATH=. .venv/Scripts/python.exe -m app.jobs.ingest_docs
```

### 4. 起两个 MCP Server + 启动服务

```bash
# 两个业务 MCP Server 各自独立进程（物流/售后，Streamable HTTP）
.venv/Scripts/python.exe -m mcp_servers.logistics_server   # 127.0.0.1:8100
.venv/Scripts/python.exe -m mcp_servers.aftersale_server   # 127.0.0.1:8101

# 客服服务
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 或 make dev
```

浏览器打开：

| 页面 | 地址 | 说明 |
|------|------|------|
| 聊天 | <http://127.0.0.1:8000/> | 主聊天页 |
| 知识库后台 | <http://127.0.0.1:8000/kb> | 建库 / 检索 / 作业 |
| 总览 | <http://127.0.0.1:8000/admin> | 聚合首页 |
| 待审队列 | <http://127.0.0.1:8000/review> | 数据飞轮人工审核 |
| 主题分布 | <http://127.0.0.1:8000/topics> | 主题分类统计 |
| 验收 | <http://127.0.0.1:8000/acceptance> | 主题分类器验收报告 |

### 5.（可选）Langfuse 可观测性

```bash
docker compose -f langfuse/docker-compose.yml up -d   # UI http://localhost:3000
# 在 .env 填 LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY / LANGFUSE_HOST
```

配好 env 后 Graph 会自动挂 `CallbackHandler` 拿完整 trace 树；未配 secret 则静默跳过，不影响主链路。

### 6.（可选）主题分类器（ch10）

旁路批量归类，需独立推理服务。完整流水线（CPU + Ollama 生成 1700 条约 2h+，请在独立终端跑）：

```bash
# 下载 RoBERTa-wwm-ext（HF 被墙，走 ModelScope 镜像）
PYTHONPATH=. .venv/Scripts/python.exe -m ml.download

# 一键：生成模拟样本 → 建集 → 训练 → 评测 → 导出 ONNX
PYTHONPATH=. .venv/Scripts/python.exe -m ml.run_pipeline

# 起独立推理服务（端口 8001）
.venv/Scripts/python.exe -m uvicorn inference.app:app --port 8001
```

分步命令与顺序见 `dev-notes/ch10.md`（生成 → 建集 → 训练 → 评测 → 导出 ONNX → 起服务 → 归类）。

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
| POST | `/api/feedback` | 👎 反馈落池（source=`user_feedback`，尽力回捞召回快照） |
| GET | `/api/review` | 列待审队列（可按 `status` 过滤） |
| GET | `/api/review/{id}` | 待审详情（缺口 + 归并原话 + 召回快照） |
| POST | `/api/review/{id}/approve` | 通过（写回知识库 + 向量化） |
| POST | `/api/review/{id}/reject` | 驳回 |
| GET | `/api/topics/distribution` | 主题分布统计 |
| GET | `/api/topics/questions?label=` | 类目问题列表 |
| GET | `/api/acceptance/*` | 主题分类器验收数据源（报告 / 数据 / 错例） |
| GET/POST | `/api/kb/*` · `/api/admin/overview` | 知识库与后台接口 |
| GET/POST | `/api/jobs/{name}/status·log·start` | 后台任务状态 / 日志 / 触发 |

## 知识库后台 / 作业运行器

`/kb` 页内置进程内作业运行器（`app/core/jobs.py` + `/api/jobs`）。因 Milvus Lite 单库单进程，作业在 uvicorn 进程内后台线程串行执行（`_START_LOCK` 防 TOCTOU + `_RUN_LOCK` 串行化），日志落 `log/jobs/`。白名单作业：

| 作业名 | 说明 |
|--------|------|
| `kb-preview` | 切分预览（dry run） |
| `kb-build` | 建库（幂等，已有块则拒绝，需 `kb-reset` 重建） |
| `kb-mine` | 从历史对话挖 QA |
| `kb-vectorize` | 向量化 pending 块 |
| `kb-reset` | 重建 Milvus 集合 |
| `flywheel` | 数据飞轮：标准化 → 查重 → 入待审队列 |
| `eval-run` | 评估流水线（检索指标落 `eval_runs`） |
| `classify-topics` | 主题批量归类（调 8001 推理服务） |

终端等价命令见 `Makefile`（`kb-preview` / `kb-build` / `kb-mine` / `kb-vectorize` / `kb-reset` / `eval-retrieval`）。

## 数据飞轮（ch09）

把「答不上」变「改进燃料」：

```
三入口(检索低置信度 / self_check 自评不足 / 用户点👎)
   → low_confidence_questions(标 source + 召回片段快照)
   → flywheel job(标准化 → 查重 → review_queue,命中累加 occurrence_count)
   → /review 待审页人工审核
   → 通过:写 knowledge_chunks(content_type=faq) + 向量化 → 再问可答对
```

`self_check` 为事后判断：证据不足则落池、不拦截已流出的答案。

## 主题分类器（ch10）

- 术语表：`ml/taxonomy.py` 17 类（单一权威源，全链路引用一份）。
- 训练数据：`ml/data/*.jsonl`（`hand_written` 真实样例 + `raw_synthetic` 大模型造样）。
- 产物：`ml/artifacts/best.pt`（权重）、`ml/artifacts/eval_report.json`（每类 P/R/F1 + 混淆矩阵）、`ml/artifacts/topic_classifier.onnx`（部署）。
- 归类：`classify-topics` job 攒未归类的低置信度问题 → 调 `http://127.0.0.1:8001/classify` → 写 `topic_classifications`（多标签，全低于阈值归「其他」）。

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

## 开发记录

各章实现细节、踩坑与关键结论见 `dev-notes/ch01.md` ～ `ch10.md`；设计文档与实现计划见 `docs/superpowers/specs/` 与 `docs/superpowers/plans/`。
