# Mew Help — 电商智能客服系统

一个本地可跑的电商售后智能客服 RAG 系统，从纯对话、Function Calling 工具链，一路做到混合检索 + 重排 + 评估体系。

## 功能

- **ch01 纯对话**：FastAPI + LangChain，`/chat` SSE 流式输出 + `/extract` 结构化信息抽取。
- **ch02 聊天前端**：`static/` 暖橙色界面，逐字渲染 SSE、多轮对话、工具调用徽章。
- **ch03 Function Calling**：5 个工具（订单/商品/物流 mock、FAQ 查询、创建工单），单轮工具链 + 消息落库 MySQL。
- **ch04 混合检索 + 重排 + 评估**：
  - Milvus Lite 内置 **BM25**（jieba 中文分词）与 **BGE-M3** dense 向量，`hybrid_search` + **RRF** 融合。
  - **bge-reranker-v2-m3** 精排（Cross-Encoder），Lost-in-the-Middle 首尾重排。
  - Query 改写 + 同义词检索侧扩写、元数据过滤。
  - 生成质量控制：引用编号溯源、自评不足拒答入池、忠实度校验。
  - 四策略评估（纯向量 / 纯 BM25 / 混合 RRF / 混合 + 精排）：Recall@K、MRR、证据覆盖度、答案覆盖度、Faithfulness、拒答率。
- **知识库管理**：`/kb` 后台页 + `/api/kb` 接口。

## 技术栈

FastAPI · LangChain · Ollama（`qwen3:8b` + `bge-m3` + `bge-reranker-v2-m3`）· MySQL 8（Docker）· Milvus Lite · SQLAlchemy · PyMilvus · sentence-transformers / torch

## 目录结构

```
app/            # FastAPI 应用（api/core/db/kb/services/tools/prompts/jobs）
knowledge/      # 知识库 Markdown 源（售后手册/商品目录/FAQ/退货政策）
evaluation/     # 评估集 + 检索/生成段评估脚本
static/         # 聊天与 /kb 前端
db/             # MySQL schema.sql
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

### 3. 起 MySQL + 灌知识库

```bash
docker compose up -d    # 启动 MySQL，首次会自动执行 db/schema.sql
# 灌入 FAQ 种子（MySQL 中文需 utf8mb4，已内置处理）
PYTHONPATH=. .venv/Scripts/python.exe -c "from app.db import SessionLocal; from app.db.seed import seed_faq; seed_faq(SessionLocal())"
# 切分 knowledge/*.md 并向量化入库（Milvus Lite）
PYTHONPATH=. .venv/Scripts/python.exe -m app.jobs.ingest_docs
```

### 4. 启动服务

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 或 make dev
```

浏览器打开 <http://127.0.0.1:8000/>。

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

## 测试

```bash
pytest
```
