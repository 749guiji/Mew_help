# 可观测性与数据飞轮(ch09) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统装 Langfuse 全链路可观测 + 意图维度 token 统计,并把「答不上」经三入口落池、标准化查重入待审队列、审核通过写回知识库的完整数据飞轮跑通。

**Architecture:** Langfuse `CallbackHandler` 挂进 chat_service 的 graph config(env 未配静默跳过);知识类闸升级成 `evidence_confidence` 三信号;新增 `self_check` 节点(事后落池不拦答案);三入口统一 `log_low_confidence` 落池带召回快照;`flywheel` job 标准化+查重+入 `review_queue`;审核通过写 `knowledge_chunks` + 向量化;`eval_run` job 落 `eval_runs` 看趋势。

**Tech Stack:** FastAPI · LangChain 1.5 / LangGraph 1.2 · Langfuse(v3, `langfuse.langchain.CallbackHandler`)· SQLAlchemy · MySQL 8 · Milvus Lite · `mcp` SDK 1.x。

**Spec:** `docs/superpowers/specs/2026-11-21-ch09-observability-flywheel-design.md`

## Global Constraints

- Langfuse v3:`from langfuse.langchain import CallbackHandler`(旧 `langfuse.callback` 已迁);`config={"callbacks":[handler], "metadata":{...}}` 挂载;`langfuse_user_id`/`langfuse_session_id`/`langfuse_tags` 元数据键。v3 回调仍是同步实现,async 图有阻塞风险——验证 trace 完整性,不行则 `@observe` 包 graph 入口兜底。
- **Langfuse 未配置时静默跳过**,任何路径不因 Langfuse 崩。
- evidence_confidence 阈值**必须用 ch04 评估集校准**,不拍脑袋;闸位置不动(知识检索后、agent 前)。
- self_check 语义:**事后判断 + 落池,不拦已流出的答案**(用户拍板 A)。
- 三入口 source 枚举:`retrieval_low_conf` / `self_check` / `user_feedback`(ch04 已有)。
- 查重命中 → `occurrence_count += 1` + 回写 `matched_review_id`,**不新建行**。
- `review_queue` / `eval_runs` / `low_confidence_questions` 加列 DDL 逐字进 `db/schema.sql`(含 `SET NAMES utf8mb4`)。
- 飞轮/评估走 `app/core/jobs.py` 的 `JOB_REGISTRY`(手动触发);审核通过走 ch03 的 `knowledge_chunks` 落库 + `vectorize_pending`。

## Review Focus

1. **Langfuse 未配置不崩**:env 缺 secret 时整条链路照常、无回调。→ Task 5 加测试。
2. **查重命中却新建重复行**:同义问题应累加不新建。→ Task 6 加测试(fake 查重返回命中)。
3. **👎 落池时快照重跑检索失败**:检索异常应快照置 None 不崩。→ Task 4 加测试。
4. **审核通过写库但向量化失败**:chunk 应留 pending 不丢,下次 vectorize 可重试。→ Task 7 加测试。
5. **evidence_confidence 对空/单 chunks 边界**:空检索应拒答、单条证据不误判。→ Task 2 加测试。

---

### Task 1: DB 模型 + DDL

**Files:**
- Modify: `app/db/models.py`(加 `ReviewQueue`、`EvalRun`;`LowConfidenceQuestion` 加 `retrieved_chunks`/`matched_review_id`)、`db/schema.sql`(逐字加用户 DDL)
- Test: `tests/test_review_models.py`(新建)

**Interfaces:**
- Produces:`ReviewQueue(id, normalized_question, ai_suggested_answer, occurrence_count, review_status, approved_answer, created_at, updated_at)`、`EvalRun(id, triggered_by, dataset_size, metrics, created_at)`、`LowConfidenceQuestion.retrieved_chunks(JSON, nullable)` + `matched_review_id(BIGINT FK→review_queue, ON DELETE SET NULL, index)`。

- [ ] **Step 1: 写失败测试** `tests/test_review_models.py`

```python
def test_review_queue_and_eval_run_models(session):
    # session 走 conftest SQLite;Base.metadata.create_all 后
    rq = ReviewQueue(normalized_question="运费怎么算", ai_suggested_answer="满99包邮", occurrence_count=3)
    session.add(rq); session.commit(); session.refresh(rq)
    assert rq.review_status == "待审" and rq.occurrence_count == 3

def test_low_confidence_question_new_columns(session):
    q = LowConfidenceQuestion(raw_question="运费?", source="retrieval_low_conf", retrieved_chunks=[{"score": 0.9, "answer": "满99包邮"}], matched_review_id=None)
    session.add(q); session.commit()
    assert q.retrieved_chunks[0]["score"] == 0.9
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_review_models.py -v` → FAIL
- [ ] **Step 3: 实现** models.py 加两模型 + LowConfidenceQuestion 两列;`db/schema.sql` 逐字加 `review_queue`/`eval_runs` DDL + `ALTER TABLE low_confidence_questions ADD COLUMN retrieved_chunks JSON ... , ADD COLUMN matched_review_id BIGINT ... ADD KEY ... ADD CONSTRAINT fk_lcq_review ...`(按用户给定)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_review_models.py -v` → PASS;存量库手动 ALTER(`review_queue`/`eval_runs` 由 migrate 建表,`low_confidence_questions` 加列手动 ALTER)。
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): review_queue/eval_runs 模型 + low_confidence_questions 加列"`

---

### Task 2: evidence_confidence 闸

**Files:**
- Create: `app/graph/confidence.py`
- Modify: `app/graph/nodes.py`(`gate_node` 用 evidence_confidence、`fallback_node` 落池带快照)
- Test: `tests/test_confidence.py`(新建)

**Interfaces:**
- Consumes:`ScoredChunk(score, answer, ...)`。
- Produces:`compute_evidence_confidence(chunks) -> dict{confidence: float, signals: dict}`,signals 含 `top1_score`/`top1_top2_gap`/`effective_count`。阈值从 `get_settings()` 或模块常量读。

`gate_node`:检索后算 evidence_confidence;低于阈值 → `fallback`(兜底话术 + `log_low_confidence(source="retrieval_low_conf", retrieved_chunks=快照, reason=signals)`)。

- [ ] **Step 1: 写失败测试** `tests/test_confidence.py`

```python
def test_empty_chunks_low_confidence():
    r = compute_evidence_confidence([])
    assert r["confidence"] == 0.0

def test_single_strong_hit_high_confidence():
    # 一条 score 0.9 的证据 → confidence 高、effective_count==1

def test_gap_detected():
    # Top1 0.9 / Top2 0.2(分差大)→ 信号里 top1_top2_gap 大
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_confidence.py -v` → FAIL
- [ ] **Step 3: 实现** `compute_evidence_confidence`(三信号加权,权重初值 + 常数占位);`gate_node`/`fallback_node` 接线。**阈值/权重留到 Task 10 用评估集校准,先跑通逻辑。**
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_confidence.py tests/test_graph.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): evidence_confidence 三信号闸 + fallback 落池快照"`

---

### Task 3: self_check 节点

**Files:**
- Modify: `app/graph/nodes.py`(加 `self_check_node`)、`app/graph/build.py`(知识类 `agent → self_check → log`)
- Test: `tests/test_self_check.py`(新建)

**Interfaces:**
- Consumes:`self_check(llm, question, evidence)`(已有)、`log_low_confidence`、`state["resolved_query"]`/`state["retrieval_chunks"]`/agent 最后一条 AIMessage 内容。
- Produces:`async self_check_node(state) -> dict`:取 agent 末条答案 + 证据,`self_check` 判 sufficient;不足→落池(source="self_check") 并返回 `{"visited":["self_check"]}`(**不改 messages,不拦答案**);够→仅 `{"visited":["self_check"]}`。

- [ ] **Step 1: 写失败测试** `tests/test_self_check.py`(monkeypatch `nodes.self_check` 返回 insufficient)

```python
def test_insufficient_logs_to_pool(monkeypatch):
    # fake self_check 返回 SelfCheckResult(False, "证据不足")+ monkeypatch log_low_confidence 记录
    # → 调用了一次,source=="self_check",且不追加 AIMessage

def test_sufficient_does_not_log(monkeypatch):
    # sufficient=True → 不落池
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_self_check.py -v` → FAIL
- [ ] **Step 3: 实现** `self_check_node` + `build.py` 知识类路由 `agent → self_check → log`(其余路由仍 `agent → log`)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_self_check.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): self_check 节点(事后落池不拦答案)"`

---

### Task 4: 👎 反馈落池

**Files:**
- Modify: `app/services/generation_quality.py`(`log_low_confidence` 加 `retrieved_chunks` 参数)、`app/api/routes/feedback.py`(新)、`app/main.py`(注册路由)、`static/app.js`(👎 接后端)
- Test: `tests/test_feedback.py`(新建)、改 `tests/test_actions.py`(若涉及)

**Interfaces:**
- Consumes:`log_low_confidence`、`default_retrieve`(重跑检索)、`SessionLocal`。
- Produces:`POST /api/feedback`(body `{conversation_id, message}`)→ 落池 source="user_feedback",`retrieved_chunks` = 对 `message` 重跑检索的 Top 快照(异常/无果则 None);返回 `{"ok": True}`。

- [ ] **Step 1: 写失败测试** `tests/test_feedback.py`

```python
def test_feedback_logs_pool_with_snapshot(monkeypatch):
    # monkeypatch default_retrieve 返回 1 条;POST /api/feedback → low_confidence_questions 一行 source=user_feedback、retrieved_chunks 非空

def test_feedback_retrieval_failure_snapshot_none(monkeypatch):
    # default_retrieve 抛异常 → 落池成功、retrieved_chunks 为 None
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_feedback.py -v` → FAIL
- [ ] **Step 3: 实现** `log_low_confidence` 加参;`app/api/routes/feedback.py` 端点;`main.py` 注册;`static/app.js` 的 👎 click 发 `POST /api/feedback`(带当轮用户问题)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_feedback.py -v` → PASS;`node --check static/app.js`
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): 👎 反馈落池 + 重跑检索快照"`

---

### Task 5: Langfuse 集成

**Files:**
- Create: `app/observability.py`、`langfuse/docker-compose.yml`
- Modify: `app/core/config.py`(加 langfuse 三字段)、`app/services/chat_service.py`(config 挂 callback + metadata + 事后写 intent tag)、`requirements.txt`(加 `langfuse`)、`.env.example`
- Test: `tests/test_observability.py`(新建)

**Interfaces:**
- Consumes:`get_settings`。
- Produces:`get_langfuse_handler() -> CallbackHandler | None`(env 未配 secret → None);`langfuse_config(conv_id, user_id) -> dict`(含 callbacks + metadata,未启用返回空 dict)。

- [ ] **Step 1: 写失败测试** `tests/test_observability.py`

```python
def test_handler_none_when_unconfigured(monkeypatch):
    # monkeypatch delenv langfuse_secret_key → get_langfuse_handler() is None

def test_config_empty_when_unconfigured():
    # 未配置时 langfuse_config(1,"u") == {}
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_observability.py -v` → FAIL
- [ ] **Step 3: 实现** config 三字段(`langfuse_secret_key`/`langfuse_public_key`/`langfuse_host` 默认 `http://localhost:3000`);`observability.py`;chat_service `_stream_invoke` 挂 callback + metadata + 图跑完后回写 `intent:{意图}` tag;`requirements.txt` 加 `langfuse`;`langfuse/docker-compose.yml`。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_observability.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): Langfuse 集成(env+callback+intent tag+compose)"`

---

### Task 6: 飞轮流水线(标准化+查重+入队)

**Files:**
- Create: `app/jobs/flywheel.py`
- Modify: `app/core/jobs.py`(注册 `flywheel` job)、`app/prompts/system.py`(加标准化/查重 prompt)
- Test: `tests/test_flywheel.py`(新建);标注样例 `tests/data/flywheel_samples.json`

**Interfaces:**
- Consumes:`SessionLocal`、`LowConfidenceQuestion`、`ReviewQueue`、`get_llm`。
- Produces:`run_flywheel() -> dict`:处理 `matched_review_id IS NULL` 的 low_confidence 行 → 标准化 → 查重 → 命中累加/否则新建 review_queue → 回写 matched_review_id。返回 `{"processed": n, "new": x, "merged": y}`。

- [ ] **Step 1: 写失败测试** `tests/test_flywheel.py`(monkeypatch normalize/dedupe LLM 函数)

```python
def test_new_gap_creates_review_row(monkeypatch):
    # fake normalize 返回 (标准问, 示例答);fake dedupe 返回 None(不重复)
    # → review_queue 新建一行,low_confidence.matched_review_id 指过去

def test_duplicate_merges_occurrence(monkeypatch):
    # fake dedupe 返回已有 review id → occurrence_count +1,不新建行,matched_review_id 回写
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_flywheel.py -v` → FAIL
- [ ] **Step 3: 实现** `flywheel.py` + prompt + 注册 job。**非可单测部分**(标准化/查重 prompt):`tests/data/flywheel_samples.json` 标注样例,脚本跑验证。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_flywheel.py -v` → PASS;标注样例命中
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): 飞轮流水线 标准化+查重+入队"`

---

### Task 7: 审核通过 → 落库

**Files:**
- Create: `app/api/routes/review.py`
- Modify: `app/main.py`(注册)、`app/kb/ingest.py`(若需单 chunk 落库 helper)
- Test: `tests/test_review_api.py`(新建)

**Interfaces:**
- Consumes:`ReviewQueue`、`KnowledgeChunk`、`vectorize_pending`/单 chunk 落库。
- Produces:`GET /api/review`(列表)、`GET /api/review/{id}`(详情含归并原话 + 快照)、`POST /api/review/{id}/approve`(body `{approved_answer}`)→ `review_status=通过` + 写 `knowledge_chunks`(content_type="faq", vectorize_status="pending")+ 触发向量化)、`POST /api/review/{id}/reject`。

- [ ] **Step 1: 写失败测试** `tests/test_review_api.py`

```python
def test_approve_writes_chunk_and_status(sqlite_session_local):
    # 先建 review_queue 一行;approve → review_status=通过、knowledge_chunks 多一行 content_type=faq

def test_approve_vectorize_failure_keeps_pending(monkeypatch):
    # vectorize 抛异常 → chunk 仍 pending、状态仍通过(不丢数据)
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_review_api.py -v` → FAIL
- [ ] **Step 3: 实现** `review.py` + `main.py` 注册;approve 复用 ch03 落库(写 KnowledgeChunk + `vectorize_pending`,向量化失败留 pending)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_review_api.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): 审核通过写回知识库"`

---

### Task 8: 评估流水线

**Files:**
- Create: `app/jobs/eval_run.py`
- Modify: `app/core/jobs.py`(注册 `eval-run` job)
- Test: `tests/test_eval_run.py`(新建)

**Interfaces:**
- Consumes:`evaluation.run_eval` 的检索/生成段(或 `evaluation.metrics`)、`EvalRun`。
- Produces:`run_eval_and_record(triggered_by="手动", limit=None) -> dict`:跑检索段(Recall@K、MRR)+ 生成段(Faithfulness)汇总成 metrics,写 `EvalRun` 一行,返回 metrics。CLI `python -m app.jobs.eval_run --triggered-by 手动`。

- [ ] **Step 1: 写失败测试** `tests/test_eval_run.py`(monkeypatch 评估段函数返回固定 metrics)

```python
def test_run_records_eval_run(monkeypatch):
    # fake 检索/生成段返回固定分数 → eval_runs 多一行,metrics JSON 匹配
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_eval_run.py -v` → FAIL
- [ ] **Step 3: 实现** `eval_run.py` + 注册 job(复用 run_eval 的指标计算,抽成可调函数)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_eval_run.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch09): 评估流水线落 eval_runs"`

---

### Task 9: 前端待审队列页(Vibe Coding)

**Files:**
- Create: `static/review.html`、`static/review.js`
- Modify: `static/admin.html`/`admin.js`(导航加「待审队列」入口,可选)
- Test: `tests/test_static.py`(补 `/review` 静态页断言)

- [ ] **Step 1:** 按用户描述直接做:`/review` 页列标准化问题/出现次数/示例答案/状态;每行可点「通过」(填核准答案)/「驳回」;点开详情看归并进来的原话 + 召回片段快照。
- [ ] **Step 2:** `node --check static/review.js`;`pytest tests/test_static.py` 通过。
- [ ] **Step 3: 提交** `git commit -m "feat(ch09): 待审队列后台页"`

---

### Task 10: 标定 + 依赖钉版本 + 全量回归

**Files:**
- Create: `scripts/calibrate_confidence.py`
- Modify: `requirements.txt`(核对 langfuse 版本钉死)、`app/graph/confidence.py`(写入校准后的权重/阈值)、`.env.example`

- [ ] **Step 1:** `scripts/calibrate_confidence.py` 在 ch04 评估集上扫权重/阈值,输出分离度最好的一组;写回 `confidence.py`(常量或 config)。
- [ ] **Step 2:** 全量 `pytest` 绿(含 Task 1–9 新增/改写,预期 ≥ 320)。
- [ ] **Step 3:** 功能演示(验收 1–6 命令序列,写进 dev-notes finish 段):起 Langfuse + MySQL + Ollama + 两 MCP → 问知识库没有的问题→兜底+待审队列→审核通过→再问能答对;点 👎→落池;评估跑两轮看趋势;Langfuse 看 trace + 意图 token 统计。
- [ ] **Step 4: 提交** `git commit -m "chore(ch09): 标定阈值 + 钉依赖版本 + 全量回归"`
