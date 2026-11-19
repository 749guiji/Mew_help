# 设计：补全 ch03 RAG 基础 §10 后台 UI（/kb 页 + 作业运行器 + /admin）

日期：2026-11-19
范围：把文档 spec §10 描述、但项目当前只做了「简化版」的后台 UI 完整补齐。核心 RAG 链路（切分/双写/挖知识/在线检索）已实现，不在本次范围。

## 意图与原因

用户提供的 ch03 文档（实战演练 RAG 链路）的 spec §10 定义了一套完整后台：`/kb` 六块录入页、作业运行器（make 白名单 + 日志回显）、`/admin` 聚合首页。项目现状：

- 核心链路全部已实现（dev-notes ch03/ch04）。
- `/kb` 目前只有 4 块简化版（材料清单与切块预览 / 手工录入 / 向量化补 pending / 检索自测）。
- 缺：库存闸条、对话挖知识块、最近入库块、检索自测的路线选择、手工录入查重；缺作业运行器；缺 `/admin`；Makefile 只有 `dev`；`content_type` 规则在 `kb.py` 和 `ingest_docs.py` 各写一份。

本次把 §10 补齐，让建库/挖知识/向量化/检索自测全链都能在浏览器里走通。

## 目标 / 非目标 / 反目标

- 目标：Makefile `kb-*` 目标；作业运行器（白名单 + 日志回显 + 防重入 + heavy 确认）；`/kb` 补齐到六块；`/admin` 聚合首页；`sources.py` 单一来源；手工录入查重指纹。
- 非目标：不做登录鉴权/审计/多人协同（spec §2 明确不做）；不做单条改删；不引入真实 cron/调度（job 做成命令可启动）。
- 反目标：不改已稳定的核心链路接口（`query_faq` 契约、chunker、ingest、dualwrite 的对外行为）；不引入 LangGraph/Langfuse。

## 约束

- **Milvus Lite 单进程**：`data/` 下的 Milvus 库同一时刻只能一个进程打开；uvicorn 已进程内缓存 Milvus，离线 CLI 与在线服务不能同时开同一库。→ 作业运行器必须**进程内执行**，不能跑 make 子进程（否则 Milvus 冲突）。
- 嵌入 = Ollama BGE-M3（`embedding_base_url`），材料目录 = `knowledge/`（均已定，不改）。
- 检索自测路线 = 项目现状四策略 `dense / sparse / hybrid / hybrid_rerank`（反映 ch04，而非文档的 dense 单路）。
- 页面有重跑按钮、但没有 shell：作业名走白名单，argv 写死。

## 方案

### 1. `sources.py` + Makefile `kb-*` 目标

- 新建 `app/kb/sources.py`：`list_kb_materials() -> list[KbMaterial]`（name / content_type / path / chars），把 `_content_type_from_path` 收进此处；`kb.py` 与 `ingest_docs.py` 都改调它，消除重复。
- `Makefile` 增加目标（每个都包一层现有的 `python -m app.jobs.X`）：`kb-preview`（切块 dry-run）、`kb-build`（`ingest_docs`）、`kb-mine`（`qa_mining`）、`kb-vectorize`（`--reconcile`）、`kb-reset`（清库重建）、`eval-retrieval`（`evaluation/run_eval.py`）。

### 2. 作业运行器（`app/core/jobs.py` + `app/api/jobs.py`）

- `app/core/jobs.py`：`JOB_REGISTRY = {name: Job(name, run_fn, is_heavy, log_name)}`，`run_fn` 是进程内可调用（即各 `app.jobs.*` 的 `main` 逻辑抽成可复用函数，或直接包装现有 CLI `main`）。提供 `start_job(name) -> job_id`、`job_status(name)`、`job_log_tail(name)`。日志落 `log/jobs/<name>.log`。同名未结束拒绝重入；`is_heavy`（kb-mine / kb-reset）前端二次确认。
- `app/api/jobs.py`：`POST /api/jobs/{name}/start`、`GET /api/jobs/{name}/status`、`GET /api/jobs/{name}/log`。执行走进程内后台线程（`asyncio.to_thread` / `BackgroundTasks`），满足 Milvus 单进程约束。
- **关键裁定**：spec 说「命令一律走 make 子进程 + 日志落盘」，但 Milvus Lite 单进程让 Milvus-touching 的 job（kb-build/kb-mine/kb-vectorize/kb-reset）不能作为子进程与 uvicorn 并存。→ 改为**进程内注册函数 + 日志落盘**：白名单更硬（连 shell 都没有）、argv 写死、日志落盘，spec 的安全不变量全保留；make 目标仍存在供「不开 uvicorn 的终端」用，两者调同一注册函数，满足「页面与终端同一份逻辑」。

### 3. `/kb` 页补齐到六块

- `app/api/routes/kb.py` 增加：
  - `GET /api/kb/overview`：块数 / pending / Milvus 条数 / 关键条款数 / 双写是否一致（pending 数 vs Milvus 数，依赖缺失返回「读不到」）。
  - `GET /api/kb/staging`：暂存表三态计数 + 逐条。
  - `GET /api/kb/recent`：最近入库（id 倒序 + 状态）。
  - `POST /api/kb/search` 加 `strategy` 字段（dense/sparse/hybrid/hybrid_rerank），复用 `app/kb/hybrid` 的四策略函数。
  - 手工录入 `/api/kb/chunks` 加**查重指纹**（`questions + answer` 归一化），命中即跳过（幂等）。
- `static/kb.html/js/css` 增加库存闸条、对话挖知识块、最近入库块、检索自测路线选择；现有材料/录入/补向量/检索保留。

### 4. `/admin` 聚合首页

- `app/api/admin.py`（或 `app/api/routes/admin.py`）：`GET /admin`（页面）+ `GET /api/admin/overview`（每模块一张卡的读数，依赖没起只该卡报读不到）。
- `static/admin.html` + `static/admin.js`：共用导航（自带样式注入）+ 模块卡片聚合首页。当前只有 `/kb` 一个模块，卡片给「知识块数 / 待向量化 / 最近入库」几个关键数。

## 我做的关键裁定（自定，非用户拍板）

- **作业运行器进程内化**（见上），替代 spec 的 make 子进程 —— Milvus Lite 单进程约束使然。
- **检索自测路线 = 四策略**，而非文档的 dense 单路 —— 项目已进化到 ch04。
- **材料目录 `knowledge/`、嵌入 Ollama** 沿用既定，不回退到文档的 `data/kb/` + SiliconFlow。
- **手工录入查重指纹取 `questions + answer`**（spec §10.2 原话），与挖知识侧的「问法归一化」去重（`qa_extract.dedupe`）区分开，二者不混用。
- **`/admin` 只做外壳 + `/kb` 一张卡**，其余模块卡留空占位（后续章节再挂）。

## 留给 builder 的

- 改动/新增文件：`app/kb/sources.py`、`app/core/jobs.py`、`app/api/jobs.py`（或 routes/jobs.py）、`app/api/routes/kb.py`、`app/api/routes/admin.py`、`static/kb.html/js/css`、`static/admin.html/js`、`Makefile`、`app/jobs/*`（如需抽 `run_fn`）。
- 测试：`test_sources`、`test_jobs_runner`（白名单/防重入/日志/重入拒绝）、`test_kb_routes`（overview/staging/recent/search strategy/dedup）、`test_admin_api`、`test_static`（admin 页存在）；纯函数（查重指纹、双写一致判定）走 TDD，页面接线跑 `/kb` 手工走查验证。
- 验证：`pytest` 全绿；`make dev` 起应用后浏览器走 `/kb` 六块 + `/admin`；`make kb-preview/kb-build/kb-vectorize` 终端可跑；`--refusal`/评估脚本不受影响。
