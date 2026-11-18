# ch04 混合检索 + 重排 + 评估体系 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把客服系统的单路 dense 检索升级为「混合检索（dense + Milvus 原生 BM25 + RRF）→ bge-reranker-v2-m3 重排 → 带引用、会拒答的生成」，并配一套四策略分桶对比评估。

**Architecture:** 重建 Milvus 集合（text/sparse/dense + 品类/章节元数据），新增 query 理解（LLM 改写 + 同义词扩展）与 reranker，串成混合检索管线；`query_faq` 工具改返回编号证据 + 结构化引用，聊天服务做自评/拒答/入池；评估集 + 四策略检索指标 + Faithfulness 裁判出报告；前端 Vibe Coding 做可点引用 + 👍👎。

**Tech Stack:** FastAPI / SQLAlchemy / LangChain(langchain-openai + langchain-ollama) / Milvus Lite 3.2.2(pymilvus 3.0.3) / bge-m3 + qwen3:8b(Ollama) / bge-reranker-v2-m3(torch + sentence-transformers) / jieba / pytest。

**Spec:** `docs/superpowers/specs/2026-11-18-ch04-hybrid-retrieval-rerank-design.md`

## Global Constraints

- Milvus Lite 3.2.2 嵌入式：中文 analyzer 用 `jieba`（**不是** `chinese`，后者报 `unknown tokenizer type`）；dense 索引 `FLAT/COSINE`（不用 faiss AUTOINDEX）；sparse 索引 `SPARSE_INVERTED_INDEX/BM25`；BM25 Function 必须建集合时配置。
- 重排 bge-reranker-v2-m3 用 torch `CrossEncoder`，**进程内**承载；置信度 = `sigmoid(reranker logit)`（∈0-1），不是原始 cosine。
- 模型默认 `qwen3:8b`；所有 structured-output 调用必须显式 `method="function_calling"`（小模型/第三方端点会静默失败）。
- 同义词扩展只在检索侧做，**不落库拆存多份**。
- 工具超时 30s（`ToolExecutor` 默认）。
- 新表 `low_confidence_questions` / `faith_cases` 按 spec 附录 A DDL（`SET NAMES utf8mb4`；`faith_cases` 的 `bucket` 取值 `A_policy/B_model/C_colloquial/E_multi`）。
- Python 3.14；依赖 `jieba` / `torch` / `sentence-transformers`。
- 现有 dense 向量只在 Milvus、不在 MySQL；重建集合必须重嵌入。

## Review Focus

1. **中文 analyzer 名**：`analyzer_params={"type":"chinese"}` 会抛 `unknown tokenizer type`；必须 `"jieba"`。→ Task 2 测试。
2. **重排分数不是 0-1**：拒答阈值直接比较原始 reranker logit（可为负/大于 1）会误拒/漏拒；必须 sigmoid 后再比 0.5。→ Task 3 测试。
3. **structured-output 静默失败**：改写/自评/裁判若用 `method="json_schema"`（langchain 1.x 默认）在 Ollama 上不生效，返回 None；必须 `method="function_calling"`。→ Task 4 测试。
4. **重建迁移写空**：既有 `knowledge_chunks` 全 `done`，`vectorize_pending` 会因无 pending 行而不写任何东西；`--rebuild` 必须 drop 集合 + 置 pending + 重嵌入。→ Task 2 测试。
5. **同义词不可入库**：扩展只改检索查询文本；入库侧每 chunk 只存一份，不得为同义词拆多行。→ Task 4 测试。

---

## File Structure

- `app/kb/milvus_store.py`（改）：新 schema（text/sparse/dense + 元数据）+ `hybrid_search` + 单路 search + filter。
- `app/kb/ingest.py`（改）：`vectorize_pending` 写全行。
- `app/jobs/ingest_docs.py`（改）：`--rebuild`。
- `app/kb/reranker.py`（新）：`Reranker` 懒加载 CrossEncoder，`rerank` 返回 sigmoid 置信度。
- `app/kb/query_understanding.py`（新）：LLM 改写 + `SYNONYM_DICT` + `expand_synonyms`。
- `app/kb/hybrid.py`（新）：四策略 + `retrieve_with_understanding` + `ScoredChunk`。
- `app/kb/retrieval.py`（改）：`default_retrieve` 改为生产入口（返回 `list[ScoredChunk]`）。
- `app/tools/db_tools.py`（改）：`query_faq` 返回 JSON（evidence + citations），可选 `category`。
- `app/services/generation_quality.py`（新）：自评 + 拒答 + 入池。
- `app/services/chat_service.py`（改）：解析 JSON、发 `citations` SSE、自评/拒答流程。
- `app/prompts/system.py`（改）：System Prompt 扩展 + 改写/自评/裁判 prompt。
- `app/db/models.py`（改）：`LowConfidenceQuestion`、`FaithCase`。
- `db/schema.sql`（改）：两张新表 DDL。
- `app/core/config.py`（改）：新 Settings + model 默认 `qwen3:8b`。
- `requirements.txt` / `.env.example`（改）。
- `knowledge/product-catalog.md`（新）：带型号商品条目。
- `evaluation/eval_set.jsonl`（新）、`evaluation/metrics.py`（新）、`evaluation/run_eval.py`（新）。
- `static/app.js` / `index.html` / `style.css`（改）：前端。

---

### Task 1: 基建 — 依赖 + 配置 + 新表 DDL/ORM

**Files:**
- Modify: `requirements.txt`、`.env.example`
- Modify: `app/core/config.py`
- Modify: `db/schema.sql`（追加两张新表，含 `SET NAMES utf8mb4;`）
- Modify: `app/db/models.py`（新增 `LowConfidenceQuestion`、`FaithCase`）
- Test: `tests/test_config.py`、`tests/test_kb_models.py`（或 `tests/test_db.py`）

**Interfaces:**
- Produces: `Settings` 新增字段 `rerank_model: str = "BAAI/bge-reranker-v2-m3"`、`rerank_top_k: int = 10`、`hybrid_dense_top_k: int = 50`、`hybrid_sparse_top_k: int = 50`、`rerank_use_fp16: bool = True`；`model` 默认改 `qwen3:8b`。
- Produces: ORM `LowConfidenceQuestion`（`id/conversation_id/raw_question/source/reason/created_at`，`source` 用 `Enum("retrieval_low_conf","self_check","user_feedback", native_enum=False)`）、`FaithCase`（`id/eval_id/bucket/query/strategy/answer/reason/citations(JSON)/judge_model/status/seen_count/first_seen_at/last_seen_at/resolution/resolved_at`，`status` 用 `Enum("未解决","已解决","无需解决", native_enum=False)`，`eval_id` 唯一约束）。

- [ ] **Step 1: 写失败测试** — `tests/test_config.py` 断言 `get_settings()` 默认 `model=="qwen3:8b"`、`rerank_top_k==10`、`hybrid_dense_top_k==50`、`hybrid_sparse_top_k==50`、`rerank_model=="BAAI/bge-reranker-v2-m3"`；`tests/test_kb_models.py` 断言两张新表的列名、`source`/`status` 枚举值、`FaithCase.eval_id` 唯一。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — 加依赖（`jieba`、`torch`、`sentence-transformers`）到 requirements；`config.py` 加字段 + `model` 默认 `qwen3:8b`；`.env.example` 的 `MODEL` 改 `qwen3:8b`；`schema.sql` 追加两张新表 DDL；`models.py` 加 ORM。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_config.py tests/test_kb_models.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): deps + config + low_confidence/faith_cases DDL/ORM"`

---

### Task 2: Milvus 新 schema + 重建 + 入库

**Files:**
- Modify: `app/kb/milvus_store.py`（新 schema、`upsert` 全行、`search_dense`/`search_sparse`/`hybrid_search`、filter）
- Modify: `app/kb/ingest.py`（`vectorize_pending` 写全行）
- Modify: `app/jobs/ingest_docs.py`（`--rebuild`）
- Test: `tests/test_milvus_store.py`、`tests/test_ingest.py`

**Interfaces:**
- Consumes: `Settings`（Task 1 的 `embedding_dim` 等）。
- Produces:
  - `MilvusStore.ensure_collection()`：新 schema —— `id INT64 PK`、`text VARCHAR(≤2048, enable_analyzer=True, analyzer_params={"type":"jieba"})`、`sparse SPARSE_FLOAT_VECTOR`（`Function(name="text_bm25", input=["text"], output=["sparse"], type=BM25)`）、`dense FLOAT_VECTOR dim=settings.embedding_dim`、`category VARCHAR(128)`、`content_type VARCHAR(32)`、`section_path VARCHAR(512)`、`is_key_clause BOOL`。
  - `MilvusStore.upsert(rows: list[MilvusRecord]) -> None`，`MilvusRecord` = `@dataclass(id:int, text:str, dense:list[float], category:str, content_type:str, section_path:str|None, is_key_clause:bool)`。
  - `MilvusStore.search_dense(vector: list[float], top_k: int, filter: str|None=None) -> list[Hit]`（`metric_type=COSINE`）。
  - `MilvusStore.search_sparse(text: str, top_k: int, filter: str|None=None) -> list[Hit]`（`anns_field="sparse"`, `metric_type="BM25"`）。
  - `MilvusStore.hybrid_search(dense_vector: list[float], sparse_text: str, top_k: int, filter: str|None=None) -> list[Hit]`（`AnnSearchRequest`×2 + `RRFRanker`）。
  - `Hit` = `@dataclass(id:int, score:float, text:str, category:str, content_type:str, section_path:str|None)`。
  - `ingest.vectorize_pending(session, store, embedder) -> int`：每 pending 行 embed 后以 `MilvusRecord(text=category\nquestions\nanswer, ...)` 全行 upsert。
  - `ingest_docs` 新增 `--rebuild`：drop 旧集合 → `ensure_collection` → 所有 `knowledge_chunks` 置 `pending` → 走 `vectorize_pending` 重灌。

- [ ] **Step 1: 写失败测试** — `test_milvus_store.py`：①`ensure_collection` 后 schema 含 `sparse` 字段且 `Function` 存在；②`analyzer_params={"type":"chinese"}` 抛异常（Review Focus #1）；③`upsert` 全行后 `search_sparse("退货")` 命中、`search_dense` 命中、`hybrid_search` 返回融合排序；④`filter='category == "退货政策"'` 只返回该品类（Review Focus 无，属 Task 1 遗留的过滤能力）。`test_ingest.py`：`--rebuild` 后全部行 `vectorize_status=="done"` 且 Milvus 有数据（Review Focus #4）。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — 按 Produces 签名改造 store/ingest/ingest_docs；`Hit`/`MilvusRecord` 用 dataclass；`hybrid_search` 用 `AnnSearchRequest` + `RRFRanker`（照 Context7 示例 API）。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_milvus_store.py tests/test_ingest.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): Milvus BM25+hybrid schema, full-row upsert, --rebuild"`

---

### Task 3: 重排器 Reranker

**Files:**
- Create: `app/kb/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `Settings.rerank_model` / `rerank_use_fp16`。
- Produces: `Reranker`：`__init__(self, model_name: str, use_fp16: bool = True, underlying=None)`（`underlying` 注入用于测试，避免 torch 导入）；`rerank(self, query: str, docs: list[str]) -> list[float]`（懒加载 `CrossEncoder`，对 `[(query, doc) for doc in docs]` 打分，返回 `sigmoid(logit)` ∈ [0,1]，长度与 docs 一致）。
- 关键算法：`sigmoid(x) = 1/(1+exp(-x))`；`CrossEncoder` 首载延迟加载（`_load()` 惰性），`underlying` 非 None 时直接用。

- [ ] **Step 1: 写失败测试** — `test_reranker.py`：注入 fake `underlying`（返回 logit 如 `[2.0, -1.0]`），断言 `rerank` 返回 `sigmoid([2.0,-1.0])`≈`[0.88, 0.27]` 且 ∈[0,1]（Review Focus #2）；断言未调用时 `underlying` 不被加载（惰性）。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — `Reranker` 类；`rerank` 用 `underlying.predict(pairs)`（注入时）或懒加载 `CrossEncoder(model_name)`，`sigmoid` 后返回。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_reranker.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): Reranker (bge-reranker-v2-m3 CrossEncoder, lazy, sigmoid)"`

---

### Task 4: Query 理解 — LLM 改写 + 同义词扩展

**Files:**
- Create: `app/kb/query_understanding.py`
- Test: `tests/test_query_understanding.py`

**Interfaces:**
- Consumes: `Settings`、LLM（`with_structured_output(method="function_calling")`）。
- Produces:
  - `QueryUnderstanding` = pydantic `{standard_question: str, category: str | None}`。
  - `rewrite_query(llm, question: str) -> QueryUnderstanding`：用 `with_structured_output(QueryUnderstanding, method="function_calling")` 改写口语→标准问法 + 推断品类。
  - `SYNONYM_DICT: dict[str, list[str]]`（领域同义词表，如 `{"退款":["退钱","退货退款"],"运费":["邮费"],"发货":["寄出","物流"]}`）。
  - `expand_synonyms(text: str) -> str`：把命中 `SYNONYM_DICT` 键的词替换/追加为「原词 + 同义词」（仅返回文本，不落库）。

- [ ] **Step 1: 写失败测试** — `test_query_understanding.py`：①注入 fake LLM（返回结构化 `QueryUnderstanding(standard_question="手机壳如何退货", category="退货政策")`），断言 `rewrite_query` 正确透传且走 `function_calling`（fake 断言 `method`）；②`expand_synonyms("退款")` 含「退款」和「退钱」等；③入库侧不拆多份：断言 `SYNONYM_DICT` 是纯查表映射、`expand_synonyms` 不产生新 chunk（Review Focus #5，用 `chunk_markdown` 对扩展文本切分仍只产出等量 chunk）。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — `QueryUnderstanding` + `rewrite_query` + `SYNONYM_DICT` + `expand_synonyms`。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_query_understanding.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): query rewrite (LLM) + synonym expansion (retrieval-side)"`

---

### Task 5: 混合检索编排 + 检索入口重写

**Files:**
- Create: `app/kb/hybrid.py`
- Modify: `app/kb/retrieval.py`（`default_retrieve` 改为生产入口）
- Test: `tests/test_hybrid.py`、`tests/test_retrieval.py`

**Interfaces:**
- Consumes: `MilvusStore`（Task 2）、`Reranker`（Task 3）、`rewrite_query`/`expand_synonyms`（Task 4）、`Embedder`、SQLAlchemy session。
- Produces:
  - `ScoredChunk` = `@dataclass(chunk_id:int, score:float, section_path:str|None, category:str, content_type:str, answer:str, questions:str)`（answer/questions 从 MySQL `KnowledgeChunk` 回填）。
  - 四策略（eval 用，均返回 `list[ScoredChunk]`，`top_k` 默认 10）：
    - `dense_retrieve(query, *, session, store, embedder, top_k=10, filter=None)`
    - `sparse_retrieve(query, *, session, store, top_k=10, filter=None)`
    - `hybrid_retrieve(query, *, session, store, embedder, top_k=10, filter=None)`（RRF，无重排）
    - `hybrid_rerank_retrieve(query, *, session, store, embedder, reranker, top_k=10, dense_top_k=50, sparse_top_k=50, filter=None)`
  - `retrieve_with_understanding(query, *, session, store, embedder, reranker, rewrite_llm, top_k=10, filter=None) -> list[ScoredChunk]`：改写 → 同义词扩展（只作用于 BM25 文本）→ `hybrid_rerank_retrieve`（dense 用 standard_question，sparse 用扩展文本）。
  - `retrieval.default_retrieve(query) -> list[ScoredChunk]`：用 `get_settings()` + 缓存的 store/embedder/reranker/llm 调 `retrieve_with_understanding`（模块级缓存，进程内单例，避免重复开 Milvus Lite）。
- 关键算法：RRF 融合照 Context7 `AnnSearchRequest + RRFRanker`；重排对融合 Top-K 候选的 `answer` 文本打分；MySQL 回填 `ScoredChunk` 字段。

- [ ] **Step 1: 写失败测试** — `test_hybrid.py`：注入 fake store/embedder/reranker/llm/session，①四策略分别返回预期排序（dense 按 cosine、sparse 按 BM25、hybrid 按 RRF、hybrid_rerank 按 sigmoid 重排）；②`retrieve_with_understanding` 断言 BM25 用扩展文本、dense 用 standard_question；③`filter` 透传给 store。`test_retrieval.py`：`default_retrieve` 返回 `list[ScoredChunk]`（fake 组件下）。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — 按 Produces 签名实现；`retrieval.py` 改为缓存组件 + 薄封装。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_hybrid.py tests/test_retrieval.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): hybrid orchestration + four retrieval strategies"`

---

### Task 6: 生成质量控制 + 在线接线（query_faq / 自评 / 拒答 / SSE 引用）

**Files:**
- Create: `app/services/generation_quality.py`
- Modify: `app/tools/db_tools.py`（`query_faq` 返回 JSON + 可选 `category`）
- Modify: `app/services/chat_service.py`（解析 JSON、发 `citations` SSE、自评/拒答/入池）
- Modify: `app/prompts/system.py`（System Prompt 扩展 + 改写/自评/裁判 prompt）
- Test: `tests/test_generation_quality.py`、`tests/test_tools.py`、`tests/test_chat_service.py`

**Interfaces:**
- Consumes: `retrieve_with_understanding`（Task 5）、`LowConfidenceQuestion`（Task 1）。
- Produces:
  - `self_check(llm, question: str, evidence: str) -> SelfCheckResult`，`SelfCheckResult` = pydantic `{sufficient: bool, reason: str}`，用 `with_structured_output(..., method="function_calling")`。
  - `log_low_confidence(session, *, conversation_id: int|None, raw_question: str, source: str, reason: str|None) -> None`：插 `low_confidence_questions`。
  - `format_evidence(chunks: list[ScoredChunk]) -> str`：`[1] {section_path or category}: {answer}` 编号文本。
  - `build_citations(chunks: list[ScoredChunk]) -> list[dict]`：`[{n, chunk_id, section_path, answer_snippet}]`。
  - `query_faq(keyword: str, category: str|None = None) -> str`：返回 JSON 字符串 `{"refused": bool, "evidence": str, "citations": list}`；内部调 `retrieval.default_retrieve(keyword)`（Task 5 的缓存组件生产入口，非直接注入 `retrieve_with_understanding`）；检索空/`max(score)<retrieval_threshold` → `refused=true`（不调自评）。
  - `chat_service`：query_faq 结果 `refused=true` → 发固定拒答 token + `log_low_confidence(source="retrieval_low_conf")`；否则发 `{"type":"citations","citations":[...]}` SSE → 自评 `self_check` → `sufficient=false` → 拒答 + `log_low_confidence(source="self_check")` → 否则最终生成（未绑工具模型，prompt 带 evidence + 引用指示）。

- [ ] **Step 1: 写失败测试** — `test_generation_quality.py`：①`self_check` 注入 fake LLM 返回 `{sufficient:false, reason:"缺运费"}` 断言透传 + `function_calling`；②`log_low_confidence` 落一行且 `source`/`reason`/`raw_question` 正确；③`format_evidence` 编号 + section_path。`test_tools.py`：`query_faq` fake 检索返回 JSON，空结果 `refused=true`。`test_chat_service.py`：query_faq 调用后发 `citations` SSE；自评不足时发拒答 + 写池。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — 按 Produces 签名实现；`SYSTEM_PROMPT` 增加「句末 [n] 引用」「证据不足拒答」「禁止承诺到账时间」。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_generation_quality.py tests/test_tools.py tests/test_chat_service.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): generation quality (self-check, refusal, citations, low-confidence pool)"`

---

### Task 7: 评估集 + 语料补型号 + 检索段指标（Recall@K / MRR，四策略分桶）

**Files:**
- Create: `knowledge/product-catalog.md`（含 SKU-1001 手机壳等带型号条目）
- Create: `evaluation/eval_set.jsonl`
- Create: `evaluation/metrics.py`
- Create: `evaluation/run_eval.py`（检索段部分）
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 四策略（Task 5）、`MilvusStore`/`Embedder`/`Reranker`（真实组件）。
- Produces:
  - `eval_set.jsonl` 每行 `{id, bucket, query, ground_truth_chunk_ids: [int], difficulty}`，桶 `A_policy/B_model/C_colloquial/E_multi`，含带型号问法（`B_model`）。
  - `metrics.recall_at_k(hit_ids: list[int], ground_truth: list[int], k: int) -> float`
  - `metrics.mrr(hit_ids: list[int], ground_truth: list[int]) -> float`
  - `run_eval.py`：对四策略各跑一遍 eval_set，按 bucket 汇总 Recall@K/MRR，打印分桶对比表。

> 本条属「数据 + 非可单测代码」：不套 TDD，改「跑评估验证」。`metrics.py` 是纯函数，仍写单测。

- [ ] **Step 1: 写 metrics 单测** — `test_metrics.py`：`recall_at_k`/`mrr` 用固定 `hit_ids`/`ground_truth` 断言（含 0 命中、K 超界）。
- [ ] **Step 2: 跑单测通过** — `pytest tests/test_metrics.py -v`
- [ ] **Step 3: 造数据** — 写 `product-catalog.md`（≥3 条带型号条目）、`eval_set.jsonl`（40-60 条，四桶 + 难度梯度 + `ground_truth_chunk_ids`）。
- [ ] **Step 4: 实现 run_eval 检索段** — 四策略跑 eval_set，分桶输出 Recall@K/MRR 表。
- [ ] **Step 5: 验证（跑评估）** — 先 `ingest_docs --rebuild` 灌入（含 product-catalog），再 `python evaluation/run_eval.py`，确认四策略表有数字、`B_model` 桶 BM25 命中型号（验收 #1/#2）。
- [ ] **Step 6: Commit** — `git commit -m "feat(ch04): eval set + catalog + Recall@K/MRR four-strategy report"`

---

### Task 8: 生成段 Faithfulness + faith_cases 台账

**Files:**
- Modify: `evaluation/run_eval.py`（加生成段）
- Create: `evaluation/faithfulness.py`（裁判 + 入账）
- Test: `tests/test_faithfulness.py`

**Interfaces:**
- Consumes: `FaithCase`（Task 1）、`retrieve_with_understanding` + 生成（Task 6 的 prompt）。
- Produces:
  - `judge_faithfulness(llm, question: str, answer: str, citations: list[dict]) -> FaithfulnessResult`，`FaithfulnessResult` = pydantic `{faithful: bool, reason: str}`（`function_calling`）。
  - `record_faith_case(session, *, eval_id, bucket, query, strategy, answer, reason, citations, judge_model) -> None`：按 `eval_id` 唯一 upsert（已存在则更新 answer/reason/citations 快照 + `seen_count+1` + `last_seen_at`；`status` 由「已解决/无需解决」复发退回「未解决」）。
  - `run_eval.py` 生成段：对生产策略（hybrid+rerank）跑生成 → 裁判判 faithful → 不 faithful 的写 `faith_cases` → 汇总 Faithfulness 率入报告。

> 本条含裁判 prompt（非可单测）+ 台账 upsert 逻辑（可单测）：prompt 用评估集验证，`record_faith_case` 写单测。

- [ ] **Step 1: 写单测** — `test_faithfulness.py`：`record_faith_case` 首次插入 / 重复 `eval_id` 更新 `seen_count`+`last_seen_at` / 复发状态退回「未解决」（fake session 或内存）。
- [ ] **Step 2: 跑单测通过** — `pytest tests/test_faithfulness.py -v`
- [ ] **Step 3: 实现** — `judge_faithfulness` + `record_faith_case`；`run_eval.py` 生成段 + Faithfulness 汇总。
- [ ] **Step 4: 验证（跑评估）** — `python evaluation/run_eval.py` 出 Faithfulness 表，`faith_cases` 有写入（验收 #3 支撑）。
- [ ] **Step 5: Commit** — `git commit -m "feat(ch04): faithfulness judge + faith_cases ledger"`

---

### Task 9: 前端 — 可点引用 + 👍👎 反馈（Vibe Coding）

**Files:**
- Modify: `static/app.js`、`static/index.html`、`static/style.css`

**要求（按用户描述直接做，不套 TDD/brainstorm/review）：**
- 回答里的 `[1][2]` 渲染为可点链接；点开显示来源 chunk 的 `section_path` + 原文（用 SSE `citations` 事件里的 `n/chunk_id/section_path/answer_snippet` 映射）。
- 每条 assistant 回答左下角挂 👍/👎；点击点亮所选、显示「已反馈」、一次性锁定；纯前端，不落库。

- [ ] **Step 1: 实现前端** — 改 `app.js` 处理 `citations` 事件 + 渲染可点引用 + 👍👎 锁定逻辑；`index.html`/`style.css` 加引用面板与反馈样式。
- [ ] **Step 2: 手测** — 起服务，发一个知识类问题，确认引用可点、面板显示来源、👍👎 锁定。
- [ ] **Step 3: Commit** — `git commit -m "feat(ch04): clickable citations + 👍👎 feedback (frontend)"`

---

### Task 10: 端到端联调 + 交付

**Files:**
- Modify: `dev-notes/ch04.md`（收尾阶段记录）
- 可能小修：联调发现的集成问题。

- [ ] **Step 1: 端到端** — `ingest_docs --rebuild` 重建灌库 → 起 uvicorn → `curl /chat` 发知识类问题验证引用/拒答/入池 → 发库外问题验证拒答 + `low_confidence_questions` 有行（验收 #3/#4）。
- [ ] **Step 2: 全量测试** — `pytest` 全绿。
- [ ] **Step 3: 交付材料** — 演示命令、测试结果、dev-notes 路径汇总。
- [ ] **Step 4: Commit** — `git commit -m "docs(ch04): dev-notes + e2e delivery"`

---

## 依赖顺序

Task 1 → {2, 3, 4} → 5 → 6 → {7, 8} → 9 → 10（其中 2/3/4 相互独立可并行；7 依赖 5 的四策略 + Task 1 的 DDL；8 依赖 6 的生成 + 7 的评估集）。
