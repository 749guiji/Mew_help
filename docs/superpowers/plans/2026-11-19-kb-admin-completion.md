# 补全 ch03 RAG 基础 §10 后台 UI — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 spec §10 的后台补齐:Makefile `kb-*` 目标、进程内作业运行器、`/kb` 六块页、`/admin` 聚合首页、`sources.py` 单一来源、手工录入查重。

**Architecture:** 作业运行器把「离线 make 目标」做成进程内注册函数(避开 Milvus Lite 单进程与 uvicorn 的冲突),日志落 `log/jobs/<name>.log`,页面轮询回显;make 目标与页面调同一注册函数。`/kb` 复用现有 in-process 的 Milvus/embedder 缓存(`app/kb/retrieval`),材料清单收进 `app/kb/sources.py`。

**Tech Stack:** FastAPI + SQLAlchemy + pymilvus(Milvus Lite)+ LangChain + 原生 JS(前端 Vibe Coding)。

**Spec:** `docs/superpowers/specs/2026-11-19-kb-admin-completion-design.md`

## Global Constraints

- **Milvus Lite 单进程**:作业一律进程内执行(uvicorn 已缓存 Milvus),不做 make 子进程。
- 材料目录 `knowledge/`、嵌入 Ollama BGE-M3(既定,不回退)。
- 检索自测路线 = `dense / sparse / hybrid / hybrid_rerank` 四策略。
- 页面有重跑按钮、无 shell:作业名白名单,argv 写死;heavy(kb-mine/kb-reset)前端二次确认;同名未结束拒绝重入。
- 手工录入查重指纹 = `questions + answer` 归一化(与挖知识侧 `qa_extract.dedupe` 的「问法归一化」区分)。
- 确定性代码走 TDD;前端走 Vibe Coding(仅 `node --check` + `test_static` 存在性断言)。
- 每任务结束提交一次。

## Review Focus

1. **作业重入**:同名作业未结束再次 start 必须拒绝(不并行)。→ Task 2 测试。
2. **白名单**:未知作业名 start/status/log 必须 404。→ Task 2/3 测试。
3. **双写一致判定**:pending 数 vs Milvus 数「一致/不一致/读不到」三态,依赖缺失不能报「对不上」。→ Task 4 测试。
4. **查重指纹**:同「问法+正文」二次录入必须跳过(幂等),不同正文的同一问法不得误杀。→ Task 4 测试。
5. **检索路线参数**:`strategy` 非法值必须回 422/400,不能静默回退。→ Task 4 测试。

---

### Task 1: `sources.py` + Makefile `kb-*` 目标

**Files:**
- Create: `app/kb/sources.py`
- Modify: `app/api/routes/kb.py`、`app/jobs/ingest_docs.py`(改用 sources 的 content_type 规则)
- Modify: `Makefile`
- Test: `tests/test_sources.py`(新建)

**Interfaces:**
- Produces: `content_type_from_path(path: Path) -> str`;`list_kb_materials(directory: Path = Path("knowledge")) -> list[KbMaterial]`;`@dataclass KbMaterial(name: str, path: Path, content_type: str, chars: int)`。

- [ ] **Step 1: 写失败测试**

`tests/test_sources.py`:
- `test_content_type_rule`:`faq` 文件名 → `"faq"`;`policy`/`退货` → `"policy"`;其余 → `"manual"`。
- `test_list_materials_reads_knowledge_dir`:用 `tmp_path` 建两个 .md,断言按名字排序、`chars` 正确、`content_type` 正确。

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_sources.py -v`(ModuleNotFoundError)

- [ ] **Step 3: 实现 `app/kb/sources.py`**

`content_type_from_path`:`"faq" in name → faq`;`"policy" in name or "退货" in name → policy`;否则 `manual`。`list_kb_materials` 用 `sorted(directory.glob("*.md"))`,读 `read_text(encoding="utf-8")` 得 chars。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: 改 `kb.py` 与 `ingest_docs.py` 复用 sources**

删两处的 `_content_type_from_path`,改 `from app.kb.sources import content_type_from_path, list_kb_materials`;`kb.py` 的 `/api/kb/materials` 改调 `list_kb_materials()` 返回 `[{name, content_type, chars}]`。运行 `pytest tests/test_kb_routes.py tests/test_ingest.py -v` 全绿。

- [ ] **Step 6: Makefile 加目标**

```make
kb-preview:   ; .venv/Scripts/python.exe -m app.jobs.ingest_docs --preview     # dry-run 切块,不写库
kb-build:     ; .venv/Scripts/python.exe -m app.jobs.ingest_docs --force
kb-mine:      ; .venv/Scripts/python.exe -m app.jobs.qa_mining
kb-vectorize: ; .venv/Scripts/python.exe -m app.jobs.ingest_docs --reconcile
kb-reset:     ; .venv/Scripts/python.exe -m app.jobs.ingest_docs --rebuild
eval-retrieval: ; .venv/Scripts/python.exe -m evaluation.run_eval
```

(`--preview` 是 Task 2 给 `ingest_docs` 新增的 dry-run 切块 flag,复用 `_run_kb_preview` 逻辑。)

- [ ] **Step 7: Commit** — `git add app/kb/sources.py app/api/routes/kb.py app/jobs/ingest_docs.py Makefile tests/test_sources.py && git commit -m "feat(kb): sources.py 材料单一来源 + Makefile kb-* 目标"`

---

### Task 2: 作业运行器核心 `app/core/jobs.py`

**Files:**
- Create: `app/core/jobs.py`
- Modify: `app/main.py`(若需初始化 log 目录,无需则可跳过)
- Test: `tests/test_jobs.py`(新建)

**Interfaces:**
- Consumes: `app.kb.retrieval._get_store`、`app.kb.embedding.Embedder`、`app.db.SessionLocal`、`app.kb.ingest`、`app.kb.sources`、`app.kb.qa_extract`。
- Produces:
  - `@dataclass Job(name: str, run_fn: Callable[[], dict], is_heavy: bool = False)`
  - `JOB_REGISTRY: dict[str, Job]`(键 `kb-preview/kb-build/kb-mine/kb-vectorize/kb-reset`)
  - `start_job(name: str) -> str | None`(未知名抛 `KeyError`;运行中返回 `None`)
  - `job_status(name: str) -> dict`(`{"running": bool, "done": bool, "error": str|None, "log": str}`)
  - `job_log_tail(name: str, n_lines: int = 200) -> str`

- [ ] **Step 1: 写失败测试**

`tests/test_jobs.py`(注入一个 fake job 到 registry 副本测 runner 机制,不碰真实 Milvus/MySQL):
- `test_start_job_unknown_name_raises`:未知名 → `KeyError`。
- `test_reentrancy_rejected`:fake job 用 `threading.Event` 阻塞;第一次 start 返回非 None,第二次 start 返回 `None`;`event.set()` 后等其结束,第三次 start 返回非 None。
- `test_job_log_tail_returns_tail`:fake job 往 `log/jobs/<name>.log` 写 5 行,`job_log_tail(name, n_lines=3)` 返回后 3 行。

- [ ] **Step 2: 运行确认失败** → `pytest tests/test_jobs.py -v`(ModuleNotFoundError)

- [ ] **Step 3: 实现 `app/core/jobs.py`**

runner:模块级 `_STATE: dict[str, dict]`(running/done/error/thread),`_LOG_DIR = Path("log/jobs")`(mkdir parents exist_ok)。`start_job` 检查 `_STATE[name]["running"]`,起 `threading.Thread(target=_wrap, args=(name,), daemon=True)`;`_wrap` 把 `sys.stdout/stderr` 重定向到 `<log_dir>/<name>.log`(追加),调 `run_fn()`,记 done/error。`job_status` 读 `_STATE` + `job_log_tail`。注册函数(进程内,复用缓存):`_run_kb_preview`(遍历 materials → `chunk_markdown` 计数,返回 `{files, chunks, key_clauses}`)、`_run_kb_build`(遍历 materials → `ingest_document` → 返回 `{chunks}`)、`_run_kb_mine`(建 chain → `mine_conversations`)、`_run_kb_vectorize`(`vectorize_pending`)、`_run_kb_reset`(`rebuild`)。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/core/jobs.py tests/test_jobs.py && git commit -m "feat(kb): 进程内作业运行器(白名单/防重入/日志落盘)"`

---

### Task 3: 作业 API `app/api/jobs.py`

**Files:**
- Create: `app/api/jobs.py`
- Modify: `app/main.py`(include jobs router)
- Test: `tests/test_jobs_api.py`(新建,用 TestClient + monkeypatch jobs runner)

**Interfaces:**
- Consumes: `app.core.jobs`(Task 2)。
- Produces: `POST /api/jobs/{name}/start`、`GET /api/jobs/{name}/status`、`GET /api/jobs/{name}/log`;`is_heavy` 只通过 status 透传给前端(前端二次确认)。

- [ ] **Step 1: 写失败测试**

`tests/test_jobs_api.py`(monkeypatch `app.core.jobs.start_job/job_status/job_log_tail` 返回固定值):
- `test_start_unknown_job_404`:未知名 → 404。
- `test_start_known_job_returns_job_id`:fake start 返回 `"j1"` → 200 `{job_id}`。
- `test_status_and_log_endpoints`:fake status/log → 200 返回对应 JSON/文本。

- [ ] **Step 2: 运行确认失败** → FAIL(路由不存在)

- [ ] **Step 3: 实现 `app/api/jobs.py` + main.py include**

`APIRouter()`;`POST /api/jobs/{name}/start` 捕获 `KeyError` → 404,`None` → 409(已在运行),否则 200 `{job_id}`;`GET .../status` 捕获 `KeyError` → 404,返回 `job_status(name)`;`GET .../log` 返回 `PlainTextResponse(job_log_tail(name))`。`app/main.py` 加 `from app.api import jobs` + `app.include_router(jobs.router)`。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/api/jobs.py app/main.py tests/test_jobs_api.py && git commit -m "feat(kb): /api/jobs 作业接口(发起/状态/日志尾)"`

---

### Task 4: `/kb` 后端增强 `app/api/routes/kb.py` + 查重

**Files:**
- Modify: `app/api/routes/kb.py`
- Modify: `app/kb/ingest.py`(加 `chunk_fingerprint` + `find_duplicate_chunk`)
- Test: `tests/test_kb_routes.py`(扩)、`tests/test_ingest.py`(扩)

**Interfaces:**
- Consumes: `app.kb.hybrid`(四策略)、`app.kb.milvus_store`(count)、`app.db.models`、`app.kb.ingest`。
- Produces:
  - `GET /api/kb/overview` → `{"chunks": int, "pending": int, "milvus": int|None, "key_clauses": int, "consistent": bool|None}`(Milvus 读不到时 `milvus/consistent` 为 None)
  - `GET /api/kb/staging` → `{"extracted": int, "kept": int, "discarded": int, "rows": [...]}`
  - `GET /api/kb/recent?limit=20` → `[{id, category, questions, answer, content_type, vectorize_status, created_at}]`
  - `POST /api/kb/search` 加 `strategy: str = "hybrid_rerank"`(校验白名单)
  - `POST /api/kb/chunks` 加查重:`chunk_fingerprint(questions, answer) -> str`、`find_duplicate_chunk(session, fingerprint) -> int | None`,命中返回 409 `{"duplicate": true, "id": existing_id}`

- [ ] **Step 1: 写失败测试**

- `tests/test_ingest.py` 加:`test_chunk_fingerprint_same_qa_equal`(同问法+正文指纹相同)、`test_chunk_fingerprint_diff_answer_diff`(不同正文指纹不同)。
- `tests/test_kb_routes.py` 加(monkeypatch `_get_store`/`default_retrieve`/repository):`test_search_rejects_bad_strategy`(非法值 422)、`test_chunks_dedup_returns_409`(重复指纹 409)、`test_overview_consistent_true_when_counts_match`(pending==milvus 且都非 None → consistent True)。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现后端 + 查重**

`chunk_fingerprint` = `qa_extract.normalize(questions) + "|" + qa_extract.normalize(answer)`(复用归一化,小写去标点 trim);`find_duplicate_chunk` 查 `KnowledgeChunk` 全表算指纹比对。`/api/kb/overview` 用 `_get_store().client.query` 读 Milvus 计数,`try/except` 置 None。`/api/kb/search` 按 `strategy` 分派到 `hybrid.dense_retrieve/sparse_retrieve/hybrid_retrieve/hybrid_rerank_retrieve`。`/api/kb/staging`、`/api/kb/recent` 走 `SessionLocal` 读 `QaExtractionStaging`/`KnowledgeChunk`。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/api/routes/kb.py app/kb/ingest.py tests/test_kb_routes.py tests/test_ingest.py && git commit -m "feat(kb): /kb 后端 overview/staging/recent + 检索四策略 + 手工录入查重"`

---

### Task 5: `/kb` 前端补齐六块(Vibe Coding)

**Files:**
- Modify: `static/kb.html`、`static/kb.js`、`static/kb.css`
- Test: `tests/test_static.py`(扩)

**Interfaces:**
- Consumes: `/api/kb/overview|staging|recent|search(strategy)|chunks(去重)|materials|preview`、`/api/jobs/*`。

- [ ] **Step 1: 加库存闸条 + 对话挖知识 + 最近入库 + 检索路线选择**

`kb.html` 顶部加「库存闸条」区(块数/待向量化/Milvus/关键条款/双写一致 + `overview` 轮询);新增「对话挖知识」区(staging 三态 + `kb-mine` 按钮,heavy 二次确认)、「最近入库」区、检索自测加 `strategy` 下拉(dense/sparse/hybrid/hybrid_rerank);「建库材料」区加 `kb-preview`/`kb-build` 按钮;「向量化」区加 pending/done/Milvus 三数 + `kb-vectorize`/`kb-reset` 按钮(heavy 确认)。`kb.js` 对应 fetch `/api/jobs/*` 并轮询日志尾。

- [ ] **Step 2: 校验**

Run: `node --check static/kb.js` → 无语法错误;`pytest tests/test_static.py -v` → 通过(kb 页含 6 个 section)。

- [ ] **Step 3: Commit** — `git add static/kb.html static/kb.js static/kb.css tests/test_static.py && git commit -m "feat(kb): /kb 页补齐六块(闸条/挖知识/最近入库/四策略检索)"`

---

### Task 6: `/admin` 聚合首页

**Files:**
- Create: `app/api/admin.py`
- Create: `static/admin.html`、`static/admin.js`
- Modify: `app/main.py`(include admin router + 确保 /admin 由 admin.py 返回)
- Test: `tests/test_admin_api.py`(新建)、`tests/test_static.py`(扩)

**Interfaces:**
- Consumes: `app.api.routes.kb` 的 overview 读数(或直接查 repository)。
- Produces: `GET /admin`(FileResponse admin.html);`GET /api/admin/overview` → `{"cards": [{"key": "kb", "title": "知识库", "metrics": {...}, "error": str|None}]}`(依赖没起只该卡报 error)。

- [ ] **Step 1: 写失败测试**

`tests/test_admin_api.py`:`test_admin_overview_returns_kb_card`(monkeypatch 读数,返回含 `kb` 卡)、`test_admin_page_served`(`/admin` 200 且内容含 `admin`)。`test_static.py` 扩:admin.html 存在。

- [ ] **Step 2: 运行确认失败** → FAIL

- [ ] **Step 3: 实现 admin.py + admin.html/js + main.py**

`admin.py` 返回 `/admin` 页面 + `/api/admin/overview`(每卡 try/except 独立,`error` 只影响本卡)。`admin.html` 用 `admin.js` 注入共用导航(顶部「知识库 / 后台」),卡片给「知识块数/待向量化/最近入库」。`main.py` include `admin.router`。

- [ ] **Step 4: 运行确认通过** → PASS

- [ ] **Step 5: Commit** — `git add app/api/admin.py static/admin.html static/admin.js app/main.py tests/test_admin_api.py tests/test_static.py && git commit -m "feat(kb): /admin 聚合首页(模块卡片 + 共用导航)"`

---

### 最终验证

- `pytest -q` 全绿。
- `make dev` 起服务 → 浏览器 `/kb` 六块走查(材料清单/手工录入去重/检索四策略/挖知识作业/向量化补 pending/最近入库)、`/admin` 卡片显示;`make kb-build/kb-vectorize` 终端可跑(不开 uvicorn 时)。
