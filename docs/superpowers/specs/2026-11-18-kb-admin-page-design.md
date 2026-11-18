# KB 录入页（admin 调试页）设计

> 2026-11-18 · 待用户审批

## Intent & Why

现有知识库建库是纯 CLI（`ingest_docs`），没有可视化的录入/调试界面。加一个本地开发用 `/kb` 页，让操作者能看材料、预览切块、手工补录、触发向量化、自测检索，不用敲命令。

## Goals

1. `/kb` 页：材料清单（`knowledge/*.md` 列表）+ 切块预览。
2. 手工录入：单条 QA（category/questions/answer）→ 落成 pending chunk。
3. 向量化补 pending：按钮触发 `vectorize_pending`。
4. 检索自测：输入 query → `default_retrieve` 结果展示（含分数/章节路径/答案）。
5. `make dev` 起服务。

## Non-goals

- 不做文件上传/文件管理器（材料就是 `knowledge/*.md` 文件；手工录入是单条 QA）。
- 不做鉴权（本地开发工具）。
- 不做 chunk 编辑/删除/合并（只录入 + 预览 + 检索）。

## Approaches（用户拍板）

1. 手工录入 = 单条 QA 表单（不是贴 markdown 整批入库）。
2. 材料清单 = `knowledge/*.md` 源文件列表（不是 DB chunk 列表）。
3. `make dev` = 只起 uvicorn `--reload`（不打包 docker/migrate/rebuild）。

## Design

### 后端（复用 ch04 现有能力，不新增依赖）

- 路由 `GET /kb` → 返回 `static/kb.html`（`FileResponse`）。
- `GET /api/kb/materials` → 列出 `knowledge/*.md`（文件名 + 按文件名推的 content_type + 字符数）。
- `GET /api/kb/preview?path=<name>` → 对该文件跑 `chunk_markdown` → 返回 Chunk 列表（category/questions/answer/section_path/content_type/is_key_clause）。
- `POST /api/kb/chunks` → body `{content_type, category, questions, answer}` → `insert_chunks` 落成 pending 行 → 返回新 chunk id。
- `POST /api/kb/vectorize` → `vectorize_pending` → 返回补了几个。
- `POST /api/kb/search` → `default_retrieve(query)` → 返回 top-K `ScoredChunk`。

### 前端

- `static/kb.html` + `static/kb.js`（复用现有 `style.css`）：四个区块（材料清单/切块预览、手工录入、补 pending、检索自测），纯静态 + fetch 上述端点。

### Makefile

- `dev`：`.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`

### 关键点

- 所有 `/api/kb` 端点复用 uvicorn 进程内缓存的 store/embedder/reranker；补 pending 用 `vectorize_pending`（同进程 Milvus，不与离线 CLI 抢锁——前提是别在服务运行时另开 `ingest_docs`）。
- 手工录入的 chunk 落 `pending`，需再点「补 pending」才向量化进 Milvus。
- 预览/检索的分数直接复用现有 `ScoredChunk.score`。

## 留给 builder 的

- 端点/页面的精确字段命名、HTML/CSS/JS 结构、错误处理细节、测试策略（端点用 fake store/embedder/reranker 单测）。
