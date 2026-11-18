# KB 录入页 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 加一个本地 `/kb` 知识库录入/调试页：材料清单与切块预览、单条 QA 手工录入、补 pending 向量化、检索自测，外加 `make dev` 起服务。

**Architecture:** 复用 ch04 现有能力（`chunk_markdown` / `insert_chunks` / `vectorize_pending` / `default_retrieve`），加一个 `/kb` 页面路由 + 5 个 `/api/kb/*` 端点 + 纯静态前端，不新增依赖。

**Tech Stack:** FastAPI / SQLAlchemy / 现有 kb 模块 / 原生 HTML+JS。

**Spec:** `docs/superpowers/specs/2026-11-18-kb-admin-page-design.md`

## Global Constraints

- 复用 uvicorn 进程内缓存的 store/embedder/reranker；补 pending 用 `vectorize_pending`（同进程 Milvus）。
- 手工录入的 chunk 落 `pending`，不自动向量化。
- 预览用 `chunk_markdown(text, content_type, max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap_chars)`。
- 检索用 `default_retrieve(query)`（生产 hybrid+rerank 路径）。
- `make dev` 只起 uvicorn `--reload`，不打包 docker/migrate/rebuild。

## Review Focus

1. **路径穿越**：`preview?path=` 传的文件名必须限定在 `knowledge/` 目录内，不能 `../` 逃逸读任意文件。→ Task 1 测试。
2. **单进程 Milvus**：`vectorize` 端点必须复用进程内缓存的 store（`retrieval._get_store()`），不能新开 `MilvusStore`（会抢锁/重复开集合）。→ Task 1 测试。
3. **手工录入非空校验**：`questions`/`answer` 不能为空，否则落脏 chunk。→ Task 1 测试。

---

### Task 1: 后端 `/kb` 页 + `/api/kb/*` 端点 + Makefile

**Files:**
- Create: `app/api/routes/kb.py`
- Modify: `app/main.py`（include kb router）
- Create: `Makefile`
- Test: `tests/test_kb_routes.py`

**Interfaces:**
- `GET /kb` → `FileResponse(static/kb.html)`（`STATIC_DIR` 复用 main 里的定义或本模块内再定义一次）。
- `GET /api/kb/materials` → `[{name, content_type, chars}]`：列 `knowledge/*.md`，content_type 按文件名推（`faq`/`policy`/`manual`，与 `ingest_docs._content_type_from_path` 一致）。
- `GET /api/kb/preview?path=<name>` → `[{category, questions, answer, section_path, content_type, is_key_clause}]`：`path` 必须 `basename` 化后落在 `knowledge/` 内，读文件跑 `chunk_markdown`。
- `POST /api/kb/chunks` body `{content_type, category, questions, answer}` → `{id}`：`insert_chunks(session, [Chunk(...)])`（`section_path=None, is_key_clause=False`），返回新 id。
- `POST /api/kb/vectorize` → `{vectorized: int}`：`vectorize_pending(session, store, embedder)`，store 用 `app.kb.retrieval._get_store()`，embedder 用 `Embedder(dim=settings.embedding_dim)`。
- `POST /api/kb/search` body `{query}` → `[{chunk_id, score, section_path, category, answer, questions}]`：`default_retrieve(query)`。
- pydantic schema：`KBChunkCreate{content_type, category, questions, answer}`（`questions`/`answer` 非空校验）、`KBSearchRequest{query}`（非空）。

- [ ] **Step 1: 写失败测试** — `test_kb_routes.py`：用 fake store/embedder（monkeypatch `retrieval._get_store`/`Embedder`）+ SQLite session；断言 ①`/kb/materials` 列出文件 ②`/kb/preview` 正常切块 + 路径穿越（`path=../.env`）返回 4xx ③`/kb/chunks` 落 pending 行 + 空 answer 返回 4xx ④`/kb/vectorize` 复用缓存 store（断言不新建）⑤`/kb/search` 透传 `default_retrieve` 结果。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现** — `kb.py` 路由 + main.py include + Makefile（`dev: .venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`）。
- [ ] **Step 4: 跑测试确认通过** — `pytest tests/test_kb_routes.py -v`
- [ ] **Step 5: Commit** — `git commit -m "feat(kb): /kb admin page backend + /api/kb endpoints + Makefile"`

---

### Task 2: 前端 `kb.html` + `kb.js`

**Files:**
- Create: `static/kb.html`、`static/kb.js`

**要求（纯静态，复用 `style.css`）：** 四个区块——①材料清单（fetch `/api/kb/materials`，点击某材料 → `/api/kb/preview?path=` 展示切块列表）②手工录入（表单：content_type 下拉 + category + questions + answer → POST `/api/kb/chunks`，成功提示 id）③补 pending（按钮 → POST `/api/kb/vectorize`，显示补了几个）④检索自测（query 输入 → POST `/api/kb/search`，展示 chunk_id/section_path/score/answer）。

- [ ] **Step 1: 实现前端** — `kb.html` + `kb.js`。
- [ ] **Step 2: 手测** — 起服务 → 开 `http://127.0.0.1:8000/kb` → 四个区块可用、无 console 报错。
- [ ] **Step 3: Commit** — `git commit -m "feat(kb): /kb admin page frontend"`

---

## 依赖顺序

Task 1 → Task 2（前端依赖后端端点）。
