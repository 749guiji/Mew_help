# ch04 · 混合检索 + 重排 + 评估体系 · 设计文档

> 2026-11-18 · 待用户审批（builder check 后）

## 1. Intent & Why

ch03 把 `query_faq` 从关键词查表升级成了 BGE-M3 + Milvus 的**单路 dense 向量检索**。这条链路有三个明确短板：

1. **漏关键词**：dense 语义向量抓不住「精确型号 / SKU / 专有名词」这类靠字面命中的问法（如「SKU-1001 手机壳」）。用户要的验收 #2 正是这条。
2. **排名粗**：Top-1 直接出答案，没有重排，边界 case 排名质量差。
3. **不可信、不可测**：答案无引用出处、检索不到时模型硬编；没有评估体系，改坏了也不知道。

本章把检索质量做上一个台阶：**混合检索（dense + Milvus 原生 BM25 + RRF）→ bge-reranker-v2-m3 重排 → 带引用、会拒答的生成**，并配一套能跑出数字的四策略对比评估。

## 2. Goals / Non-goals / Anti-goals

### Goals
1. 关键词召回落在 Milvus 原生 BM25 上（jieba 中文分词），与 dense 向量各召回 Top-50，`hybrid_search` + RRF 融合。
2. bge-reranker-v2-m3 把融合结果精排出 Top-10。
3. 支持按品类/内容类型等标量字段先过滤再检索。
4. Query 理解：口语模糊问法 LLM 改写归一 + 同义词扩展（仅检索侧，入库侧不拆存多份）。
5. 生成质量控制：答案带可溯源引用编号、检索不到/低置信度显式拒答、生成阶段自评不足则拒答并入池、System Prompt 列负面知识。
6. 评估体系：难度梯度 + ground-truth 评估集；检索段 Recall@K/MRR，生成段 Faithfulness；四策略分桶对比报告。
7. 前端：可点引用 + 👍👎 反馈（纯前端采集，Vibe Coding）。

### Non-goals
- 指代消解、多轮改写。
- 👍👎 数据飞轮落地（本章只留前端采集入口，`low_confidence_questions.source='user_feedback'` 留给 ch09）。
- `faith_cases` 的人工处置 UI（只建表 + 评估写入，处置按钮后续章节）。

### Anti-goals
- 不在入库侧把同义词拆成多份 chunk（检索侧扩展即可）。
- 证据不足时不许硬编答案（必须拒答）。
- 不承诺到账时间等负面知识（进 System Prompt 禁止清单）。

## 3. Constraints

- **Milvus Lite 3.2.2（嵌入式）**：单 data_dir 单进程，离线建库与在线服务不能同时开同一库；Windows 上 FLAT（不用 faiss AUTOINDEX）；重开集合需 `load_collection`。
- **中文 analyzer 叫 `jieba`，不叫 `chinese`**（实测 `chinese` 报 `unknown tokenizer type`，Lite 支持 `standard`/`jieba`）。需加 `jieba` 依赖。
- **BM25 必须建集合时配置**（Function + SPARSE_INVERTED_INDEX），不能后加 → 集合要重建一次，`ingest_docs` 重灌。
- **Ollama 无 `/api/rerank`**（实测 404）→ 重排用进程内 torch（用户已拍板 A）。
- **Python 3.14**：torch 2.15.0 / onnxruntime 1.32.0 有 cp314 轮子（已验证）。
- 现有工具超时 30s（本次运行已从 5s 修上来）；本地 Ollama `qwen3:8b`（LLM）+ `bge-m3`（dense 嵌入）。
- 网络：模型从 HF 拉（bge-reranker-v2-m3 约 2.3GB），国内可能要挂 `HF_ENDPOINT=https://hf-mirror.com`。

## 4. Approaches（已决策）

### 4.1 重排承载方式（用户拍板 A）
- **A. 进程内 torch（选定）**：FastAPI 懒加载 `CrossEncoder('BAAI/bge-reranker-v2-m3')`，后台线程跑。最省事、贴合本地嵌入式；代价是加 torch、首载几秒、单次重排约 1-2s。
- B. 独立 rerank 服务：解耦但多一进程、多一套部署。→ 弃。
- C. onnxruntime/fastembed：依赖轻但模型覆盖与 ONNX 转换有一次性成本。→ 弃。

### 4.2 关键词召回承载
- 唯一路径：**Milvus 原生 BM25**（用户定死「落在 Milvus 上」）。已实测 Lite 3.2.2 完整支持 `Function(BM25) + jieba + hybrid_search + RRFRanker`，无需切 Standalone。

## 5. Design

### 5.1 Milvus 集合重建 + 入库

重建集合 `knowledge`（BM25 必须建时配置，且要补标量字段）。新 schema：

| 字段 | 类型 | 用途 |
|---|---|---|
| `id` | INT64 (PK) | 对齐 `knowledge_chunks.id` |
| `text` | VARCHAR(≤2048) `enable_analyzer=True, analyzer_params={type:'jieba'}` | BM25 全文源 = `category\nquestions\nanswer` |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出（自动生成，不手写） |
| `dense` | FLOAT_VECTOR dim=1024 | bge-m3 嵌入 |
| `category` | VARCHAR | 标量过滤（req 3） |
| `content_type` | VARCHAR | 标量过滤 |
| `section_path` | VARCHAR | 溯源元数据（引用跳转） |
| `is_key_clause` | BOOL | 元数据 |

- BM25 Function：`Function(name='text_bm25', input=['text'], output=['sparse'], type=BM25)`。
- 索引：dense `FLAT/COSINE`（沿用 Windows 坑规避）、sparse `SPARSE_INVERTED_INDEX/BM25`。
- `MilvusStore.upsert` 签名从 `(id, vector)` 扩成整行（id + text + dense + 元数据），sparse 由 Function 自动派生。
- `ingest_docs` / `vectorize_pending` 同步改造：pending 行把 dense 向量 + text + 元数据一起 upsert。
- **重建迁移语义**：新增 `ingest_docs --rebuild` 标志 = drop 旧集合 → 按新 schema 重建 → 把 `knowledge_chunks` 全部行置回 `pending` → 重新嵌入（bge-m3）+ upsert。原因：既有行全是 `done`，dense 向量只在 Milvus 不在 MySQL，只跑 `vectorize_pending` 会因「没有 pending 行」而什么都不写。普通 `ingest_docs` 与 `--reconcile` 的幂等语义保持不变（仍按 `vectorize_status` 补齐）。

### 5.2 在线检索管线（query_faq 内部）

```
原始 query
  → (1) LLM 改写归一（qwen3:8b，structured output → {standard_question, category?}）
  → (2) 同义词扩展（curated SYNONYM_DICT，仅作用于 BM25 文本查询，不落库）
  → (3) 混合检索：dense(bge-m3) Top-50 + BM25(扩展后 query) Top-50 → hybrid_search + RRFRanker
  → (4) 可选标量过滤（category 作为 filter 表达式，作用于 dense+sparse 两侧）
  → (5) bge-reranker-v2-m3 精排 Top-10
```

- 同义词扩展只改 BM25 查询文本（如「退款」→「退款 退钱 退货退款」），不改 dense 嵌入输入；dense 用改写后的 standard_question。
- 重排输入用原始 query（或 standard_question）配候选 chunk 的 `answer` 文本。
- **标量过滤来源**：改写 LLM 的 structured output 除 `standard_question` 外，额外输出可空的 `category`（从问法里推断品类，如「退货政策」→`退货政策`）；非空时作为 `filter` 传给检索。`query_faq` 工具也加可空的 `category` 入参（模型可显式传）。评估 harness 直接调检索层传 `filter`，用于分桶演示 + 验证过滤生效。
- **拒答阈值语义**：融合 + 重排后分数不再是 cosine 相似度。以 `sigmoid(reranker logit)` 作为最终置信度（∈0-1）；`max(confidence) < retrieval_threshold`（默认 0.5，可配，评估 harness 里可扫参调优）或候选为空 → 返回「无证据」标记（触发拒答，见 5.3）。

### 5.3 生成质量控制

`query_faq` 工具返回值从「纯字符串」改为 **JSON 字符串**，两段：
- `evidence`：喂给模型的编号证据（`[1] 退货政策 > 退货流程: …`）。
- `citations`：结构化来源列表 `[{n, chunk_id, section_path, snippet}]`，供前端渲染。

聊天服务对 query_faq 结果做三件事：

1. **拒答（检索侧）**：evidence 为空 / 低置信 → 输出固定拒答话术 + 写 `low_confidence_questions`（`source='retrieval_low_conf'`）。
2. **自评（生成侧）**：一条 structured 自评 LLM 调用，问「召回的证据够不够回答这个问题？」→ `{sufficient: bool, reason}`；不够 → 拒答 + 写 `low_confidence_questions`（`source='self_check'`）。
3. **引用**：够了 → 用未绑工具的模型做最终生成，System Prompt 指示「句末用 [n] 引用来源」「证据不足就拒答」「禁止承诺到账时间」；同时 SSE 新增 `citations` 事件下发结构化来源。

- 新表 `low_confidence_questions`（DDL 用户已给）：`conversation_id / raw_question / source / reason / created_at`。
- 新表 `faith_cases`（DDL 用户已给）：忠实度裁判判出的编造个案台账，评估阶段写入。

### 5.4 评估体系

- **评估集**：`evaluation/eval_set.jsonl`，每条 `{id, bucket, query, ground_truth_chunk_ids, difficulty}`。桶对齐 faith_cases 的 `A_policy / B_model / C_colloquial / E_multi`。带难度梯度。
- **语料补型号**：现有 KB 无型号，新增一份 `product-catalog.md`（含 SKU-1001 手机壳等条目），供 `B_model` 桶 + 验收 #2 用，`ingest_docs` 一并灌入。
- **检索段指标**：Recall@K、MRR。对四策略（纯 dense / 纯 BM25 / 混合 / 混合+重排）各跑一遍，按 bucket 分桶。
- **生成段指标**：Faithfulness——qwen3:8b 作裁判，判「答案是否被引用证据支撑」；判出的编造个案写 `faith_cases`。
- **报告**：`evaluation/report.md`（或 stdout），四策略 × bucket 的 Recall@K/MRR 表 + Faithfulness 表。

### 5.5 前端（Vibe Coding，不走 brainstorm/TDD/review）

- 引用编号 `[1][2]` 渲染为可点链接，点开面板显示来源 chunk 的 `section_path` + 原文。
- 每条 assistant 回答左下角挂 👍👎，点击点亮所选、显示「已反馈」、一次性锁定；纯前端，不落库。

### 5.6 配置 + 依赖

新增 Settings：`rerank_model`（默认 `BAAI/bge-reranker-v2-m3`）、`rerank_top_k=10`、`hybrid_dense_top_k=50`、`hybrid_sparse_top_k=50`、`retrieval_threshold`（沿用）、可选 `rerank_use_fp16`。

新增依赖：`jieba`、`torch`、`sentence-transformers`（CrossEncoder 承载 reranker；如遇问题可换 FlagEmbedding，模型不变）。

**模型默认值**：`config.py` 的 `model` 默认从 `qwen2.5:7b` 改为 `qwen3:8b`（同步 `.env.example`）。原因：改写 / 自评 / 忠实度裁判三处 structured-output 调用依赖 `qwen3:8b` 的 function_calling；`qwen2.5:7b` 上结构化输出会静默失败（ch01 已踩坑），且 `qwen3:8b` 是 ch01 定稿的模型。相关 `test_config` 断言同步更新。

## 6. 留给 builder 的

- 各模块的精确文件/函数签名、SSE 事件结构、同义词表内容、评估集具体题目数（约 40-60 条）。
- `query_faq` JSON 返回值与聊天服务解析的精确契约、自评链路的精确放置（工具内 vs 聊天服务）。
- 测试策略：检索/重排/拒答/自评/评估指标用 fake LLM + fake store 单测；真实联调走 e2e。

## 7. Builder check 决议（评审后自决，无「必须用户拍板」项）

builder 评审提 5 问，全部归档为「已答 / 可自决」，无需要再打扰用户的问题：

1. **拒答阈值语义**（原问：0.5 阈值在 RRF/rerank 分数上没意义）→ 已决：以 `sigmoid(reranker logit)` 为最终置信度，`max < retrieval_threshold`(默认 0.5，可配) 或空候选即拒答；阈值由评估 harness 扫参调优，不字面沿用 cosine 0.5。
2. **标量过滤来源**（原问：filter 值从哪来）→ 已决：改写 LLM 输出可空 `category` + `query_faq` 加可选 `category` 入参；评估 harness 直调检索层传 `filter` 演示并验证过滤。
3. **两张新表 DDL**（原问：faith_cases 列缺失）→ 已答：用户在需求里给了完整 DDL，本设计补入附录 A；`bucket` 是 `faith_cases` 的列（取值 `A_policy/B_model/C_colloquial/E_multi`），评估集也带同名桶字段。
4. **模型默认值**（原问：qwen2.5:7b 默认会让 structured-output 静默失败）→ 已决：`config.py` 默认改 `qwen3:8b` + `.env.example` 同步（ch01 已定稿 qwen3:8b），`test_config` 断言同步。
5. **重建迁移语义**（原问：全 done 后 `vectorize_pending` 会写空）→ 已决：新增 `ingest_docs --rebuild`（drop+重建+置 pending+重灌），普通 / `--reconcile` 幂等语义不变。

## 附录 A：两张新表 DDL（用户需求原文）

```sql
SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
```
