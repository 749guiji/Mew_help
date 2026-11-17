# RAG 知识库(ch03)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 的关键词查表换成 BGE-M3 + Milvus Lite 的向量语义检索,并建起离线文档切分、对话挖知识、MySQL+Milvus 双写落库一整套知识库管道,工具对外契约不变。

**Architecture:** 离线两条管道(文档切分、对话挖 QA)各自产出 `knowledge_chunks` 行(`vectorize_status='pending'`),共用「embed → Milvus upsert → 回填 done」的双写出口;在线 `query_faq` 改为 embed(问题)→ Milvus Top-K → 回 MySQL 取 answer。Milvus 只存 `id`(int64 主键 = MySQL id)+ `vector`(float, dim 1024, COSINE),原文权威源在 MySQL。

**Tech Stack:** Python 3.14 / FastAPI / LangChain(langchain-ollama 的 OllamaEmbeddings 指向本地 Ollama bge-m3)/ BGE-M3(1024 维)/ pymilvus + milvus-lite(嵌入式)/ SQLAlchemy + pymysql(MySQL)。

> 注:嵌入后端实测后由 `OpenAIEmbeddings`(`/v1/embeddings` 报 400)切换到 `langchain-ollama` 的 `OllamaEmbeddings`(原生 `/api/embed`),详见 dev-notes ch03 阶段三。

**Spec:** `docs/superpowers/specs/2026-11-17-ch03-rag-knowledge-base-design.md`

## Global Constraints

- 嵌入:BGE-M3,维度 **1024**,经 `langchain-ollama` 的 `OllamaEmbeddings`(`model="bge-m3"`、`base_url=embedding_base_url`,默认 `http://localhost:11434` 无 `/v1`)。
- 向量库:**Milvus Lite 3.x**(纯 Python,Windows 可跑;2.x 不支持 Windows)。Windows 上 pip 会跳过 pymilvus 的 `milvus-lite` extra,**必须显式 `pip install milvus-lite`**;`requirements.txt` 里 `pymilvus` 与 `milvus-lite` 分开各写一行。`MilvusClient("./data/milvus.db")` 生成目录,`.gitignore` 掉。
- Milvus 集合 `knowledge`:只存 `id`(int64 主键,**= MySQL `knowledge_chunks.id`**)+ `vector`(FLOAT_VECTOR 1024);`metric_type="COSINE"`,`auto_id=False`。
- 向量化文本 = `category + "\n" + questions + "\n" + answer`(三者拼一段)。
- `query_faq(keyword: str) -> str` 签名与 `@tool` 名不变;查不到返回 `"未查到"`(阈值 `retrieval_threshold` 默认 0.5 以下即「未查到」)。
- Milvus Lite 单 data_dir 单进程:离线建库 CLI 与在线 FastAPI **不同时**开同一 `.db`。
- 本章只 dense 单路;不做关键词召回/混合检索/重排。
- 状态 ENUM 用英文值(`pending`/`done`、`extracted`/`kept`/`discarded`);SQL 顶部 `SET NAMES utf8mb4` 防中文 mojibake。
- 测试沿用现有 SQLite 后端(`BIGINT_UNSIGNED` 已有 sqlite variant,Enum 用 `native_enum=False`)。

## Review Focus

下面 5 类是规格隐含、但普通任务测试容易漏的输入/故障,各配一条测试钉进归属任务:

1. **重叠裁句号必须同时识别中文(。！？)与英文(.!?)句末标点;含无任何句末标点的超长段也不能死循环/产空块** → Task 2。
2. **无 `#` 标题的 Markdown、空正文 section 不能崩或产空 chunk(用 fallback category)** → Task 2。
3. **embed 输出维度必须恒为 1024,否则 Milvus 插入报维度不匹配** → Task 3(live probe 断言)+ Task 5(vectorize 断言 `len(v)==EMBEDDING_DIM`)。
4. **Milvus 集合为空时 `search` 必须返回空列表、不抛异常** → Task 6。
5. **embed/Milvus 故障时 `query_faq` 降级返回「未查到」,不把异常抛进聊天流** → Task 6。

---

### Task 1: 依赖 + 配置 + 建表 DDL + ORM 模型

**Files:**
- Modify: `requirements.txt`
- Modify: `app/core/config.py`
- Modify: `db/schema.sql`
- Modify: `app/db/models.py`
- Modify: `.gitignore`
- Test: `tests/test_kb_models.py`(新建)

**Interfaces:**
- Produces:
  - `Settings` 新增字段:`milvus_uri: str = "./data/milvus.db"`、`milvus_collection: str = "knowledge"`、`embedding_model: str = "bge-m3"`、`embedding_dim: int = 1024`、`retrieval_top_k: int = 1`、`retrieval_threshold: float = 0.5`、`chunk_max_chars: int = 800`、`chunk_overlap_chars: int = 100`。
  - ORM:`KnowledgeChunk`(表 `knowledge_chunks`)、`QaExtractionStaging`(表 `qa_extraction_staging`),字段严格对齐 `db/schema.sql` 的 DDL。`KnowledgeChunk.vectorize_status` = `Enum("pending","done", native_enum=False, server_default="pending")`;`QaExtractionStaging.status` = `Enum("extracted","kept","discarded", native_enum=False, server_default="extracted")`;自引用 FK `prev_chunk_id`/`next_chunk_id` → `knowledge_chunks.id` ON DELETE SET NULL。

- [ ] **Step 1: 装依赖并钉版本 + Milvus Lite 冒烟**

```bash
.venv/Scripts/pip install pymilvus milvus-lite
.venv/Scripts/python -c "from pymilvus import MilvusClient; c=MilvusClient('./tmp_milvus_smoke.db'); c.create_collection('smoke', dimension=2, metric_type='COSINE', auto_id=False); c.insert('smoke', [{'id': 1, 'vector': [0.1, 0.2]}]); print(c.query('smoke', filter='id == 1'))"
```
Expected: 打印出 `id=1` 的行,无异常(Milvus Lite 在 Windows 可用)。把 `pymilvus`、`milvus-lite` 的 `pip freeze` 精确版本写进 `requirements.txt`(两行分开)。

- [ ] **Step 2: 写失败测试 `tests/test_kb_models.py`**

```python
def test_knowledge_chunk_roundtrip(session):
    c = KnowledgeChunk(category="售后>运费", questions="运费怎么收", answer="满99包邮")
    session.add(c); session.commit()
    assert c.vectorize_status == "pending" and c.vector_id is None
    # 自引用指针:插两块后回填
    c2 = KnowledgeChunk(category="售后", questions="退货", answer="7天无理由")
    session.add(c2); session.commit()
    c.next_chunk_id = c2.id; c2.prev_chunk_id = c.id; session.commit()
    assert session.get(KnowledgeChunk, c.id).next_chunk_id == c2.id

def test_qa_staging_defaults(session):
    s = QaExtractionStaging(batch_no="b1", question="q", answer="a")
    session.add(s); session.commit()
    assert s.status == "extracted"
```
(session 是复用现有 `tests` 的 SQLite 会话 fixture;如无则本任务顺带加一个 `conftest` fixture。)

- [ ] **Step 3: 跑测试确认失败**

Run: `pytest tests/test_kb_models.py -v`
Expected: FAIL(ImportError `KnowledgeChunk` 未定义)。

- [ ] **Step 4: 实现 config + DDL + models**

`config.py` 加 8 个字段(见 Interfaces)。`db/schema.sql` 追加 `knowledge_chunks`、`qa_extraction_staging` 两张表 DDL(原文取自 spec §8,含 `SET NAMES utf8mb4`)。`models.py` 加两个 ORM(字段与 DDL 对齐;`BIGINT_UNSIGNED` 复用现有 variant;`TINYINT(1)` 用 `Integer` 或 `mysql.TINYINT`)。`.gitignore` 加 `data/` 与 `*.db`。

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/test_kb_models.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/core/config.py db/schema.sql app/db/models.py .gitignore tests/test_kb_models.py
git commit -m "feat: kb deps + config + DDL + ORM models (ch03)"
```

---

### Task 2: 结构感知切分器 `chunker.py`

**Files:**
- Create: `app/kb/__init__.py`
- Create: `app/kb/chunker.py`
- Test: `tests/test_chunker.py`(新建)

**Interfaces:**
- Produces:
  - `@dataclass Chunk`: `category: str`、`questions: str`、`answer: str`、`section_path: str | None`、`content_type: str`、`is_key_clause: bool`。
  - `chunk_markdown(text: str, content_type: str, *, max_chars: int = 800, overlap: int = 100, fallback_category: str = "未分类") -> list[Chunk]`。
  - 规则(spec §6.2/§6.3):按 `#` 层级建章节路径;`content_type in ("faq",)` 时 `questions` 填真实问法、`answer` 填答案;`content_type in ("policy","manual")` 时 `questions` 填本节标题、`category` 填上级标题路径(无上级则 fallback_category)。section 正文 ≤ max_chars 整块;超长按段落→句子递归切,块间 `overlap` 重叠、重叠边界裁到最近 `。！？.!?`。Markdown pipe 表格按行切,每块前缀表头行 + 表上方最近标题。

- [ ] **Step 1: 写失败测试 `tests/test_chunker.py`**(覆盖 Review Focus #1/#2)

```python
def test_policy_heading_path():
    md = "# 退货政策\n\n## 运费说明\n\n满99元包邮,不满收8元。\n\n## 七天无理由\n\n商品完好可退。"
    chunks = chunk_markdown(md, "policy")
    assert {c.category for c in chunks} == {"退货政策"}
    assert any(c.questions == "运费说明" and "满99" in c.answer for c in chunks)

def test_faq_real_question():
    md = "# 商品FAQ\n\nQ: 支持七天无理由吗?\nA: 支持。\n"
    chunks = chunk_markdown(md, "faq")
    assert any(c.questions == "支持七天无理由吗?" and c.answer == "支持。" for c in chunks)

def test_long_section_recursive_split_and_overlap():
    body = "句子一。句子二。句子三。" * 400  # 远超 max_chars
    md = f"# 手册\n\n{long(body)}"
    chunks = chunk_markdown(md, "manual", max_chars=200, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.answer.endswith(("。","!","?","！","？")) for chunk in chunks)  # 不留半截话

def test_unpunctuated_long_text_terminates():
    md = "# 手册\n\n" + ("字" * 5000)   # 无句末标点
    chunks = chunk_markdown(md, "manual", max_chars=200, overlap=50)
    assert chunks and all(c.answer for c in chunks)   # 不死循环、不产空块

def test_table_split_replicates_header():
    md = "# 运费表\n\n| 地区 | 运费 |\n| 江浙沪 | 0 |\n| 新疆 | 30 |\n| 西藏 | 40 |"
    chunks = chunk_markdown(md, "manual", max_chars=30)  # 逼按行切
    assert len(chunks) >= 3
    assert all("| 地区 | 运费 |" in c.answer for c in chunks)  # 每块复制表头

def test_no_heading_and_empty_section():
    assert chunk_markdown("", "policy") == []            # 空输入不产块
    chunks = chunk_markdown("没有标题的正文。", "policy")
    assert chunks and chunks[0].category == "未分类"      # 无标题用 fallback
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL(`chunker` 未定义)。

- [ ] **Step 3: 实现 `chunker.py`**

纯函数,不依赖 DB/LLM。先按标题层级切 section(解析 `#`、`##`…),再对每个 section 做超长递归切(段落 `\n\n` → 句子 `。！？.!?`),overlap 边界裁到最近句末标点;表格单独检测(以 `|` 起始的连续行为 pipe 表,取首行为表头)。`is_key_clause` 简单启发式:正文命中 `必须|不得|一律|免责|责任` 任一标 True,否则 False。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_chunker.py -v`
Expected: PASS(6 passed)。

- [ ] **Step 5: Commit**

```bash
git add app/kb/__init__.py app/kb/chunker.py tests/test_chunker.py
git commit -m "feat: structure-aware markdown chunker (ch03)"
```

---

### Task 3: 嵌入客户端 `embedding.py`

**Files:**
- Create: `app/kb/embedding.py`
- Test: `tests/test_embedding.py`(新建)

**Interfaces:**
- Consumes: `Settings`(Task 1)。
- Produces:
  - `class Embedder`: `__init__(self, underlying=None, dim: int = 1024)`;`embed_texts(self, texts: list[str]) -> list[list[float]]`。
  - 默认 `underlying = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.embedding_base_url)`(来自 `langchain_ollama`);`embed_texts` 调 `underlying.embed_documents(texts)` 并 `assert len(v) == self.dim`。

- [ ] **Step 1: 写失败测试 `tests/test_embedding.py`**

```python
class FakeUnderlying:
    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

def test_embed_texts_batches():
    e = Embedder(underlying=FakeUnderlying(), dim=1024)
    out = e.embed_texts(["a", "b"])
    assert len(out) == 2 and all(len(v) == 1024 for v in out)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_embedding.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 `embedding.py`**

按 Interfaces。维度不符时抛 `ValueError`(Review Focus #3 的护栏)。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_embedding.py -v`
Expected: PASS。

- [ ] **Step 5: live probe 验证真实 Ollama bge-m3(Review Focus #3)**

```bash
ollama pull bge-m3
.venv/Scripts/python -c "from app.kb.embedding import Embedder; v=Embedder().embed_texts(['你好'])[0]; print('dim=', len(v)); assert len(v)==1024"
```
Expected: `dim= 1024`。若 Ollama `/v1/embeddings` 不兼容(报 404/维度非 1024),停下反馈(不自行换模型),按设计要求评估切 `langchain-ollama`。

- [ ] **Step 6: Commit**

```bash
git add app/kb/embedding.py tests/test_embedding.py
git commit -m "feat: bge-m3 embedder wrapper (ch03)"
```

---

### Task 4: Milvus 存储 `milvus_store.py`

**Files:**
- Create: `app/kb/milvus_store.py`
- Test: `tests/test_milvus_store.py`(新建)

**Interfaces:**
- Consumes: `Settings`(Task 1)。
- Produces:
  - `class MilvusStore`: `__init__(self, uri: str, collection: str = "knowledge", dim: int = 1024)`。
  - `ensure_collection(self) -> None`:建集合(不存在时),schema 用 `create_schema(auto_id=False)` + `add_field("id", DataType.INT64, is_primary=True)` + `add_field("vector", DataType.FLOAT_VECTOR, dim=dim)`,index `AUTOINDEX` + `metric_type="COSINE"`。幂等(已存在则跳过)。
  - `upsert(self, rows: list[tuple[int, list[float]]]) -> None`:转 `[{"id": i, "vector": v} for i, v in rows]` 调 `client.upsert(collection, ...)`。
  - `search(self, vector: list[float], top_k: int = 1) -> list[tuple[int, float]]`:调 `client.search(collection, data=[vector], limit=top_k)`,返回 `[(hit["id"], hit["distance"]) ...]`(COSINE 距离即相似度,高者更似)。

- [ ] **Step 1: 写失败测试 `tests/test_milvus_store.py`**(真实 Milvus Lite 临时目录)

```python
def test_upsert_search_and_idempotency(tmp_path):
    store = MilvusStore(str(tmp_path / "kv.db"), collection="knowledge", dim=2)
    store.ensure_collection(); store.ensure_collection()   # 幂等不炸
    store.upsert([(1, [1.0, 0.0])])
    hits = store.search([1.0, 0.0], top_k=1)
    assert hits and hits[0][0] == 1 and hits[0][1] > 0.9
    store.upsert([(1, [0.0, 1.0])])   # 同 id 覆盖,不新增
    hits2 = store.search([0.0, 1.0], top_k=5)
    assert [h[0] for h in hits2] == [1]   # 仍只有一条

def test_search_empty_collection(tmp_path):
    store = MilvusStore(str(tmp_path / "empty.db"), dim=2)
    store.ensure_collection()
    assert store.search([1.0, 0.0], top_k=1) == []   # Review Focus #4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_milvus_store.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 `milvus_store.py`**

按 Interfaces。`ensure_collection` 用 `client.has_collection(collection)` 判存在;`search` 结果里 `distance` 字段即 COSINE 相似度。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_milvus_store.py -v`
Expected: PASS(2 passed)。

- [ ] **Step 5: Commit**

```bash
git add app/kb/milvus_store.py tests/test_milvus_store.py
git commit -m "feat: milvus-lite store (ch03)"
```

---

### Task 5: 双写落库 + 幂等重跑 `ingest.py` + CLI

**Files:**
- Create: `app/kb/ingest.py`
- Create: `app/jobs/__init__.py`
- Create: `app/jobs/ingest_docs.py`
- Test: `tests/test_ingest.py`(新建)

**Interfaces:**
- Consumes: `Chunk`/`chunk_markdown`(Task 2)、`Embedder`(Task 3)、`MilvusStore`(Task 4)、`KnowledgeChunk`/`Faq`(Task 1)。
- Produces:
  - `insert_chunks(session, chunks: list[Chunk]) -> list[int]`:逐块转 `KnowledgeChunk`(`category/questions/answer/section_path/content_type/is_key_clause`,默认 `vectorize_status='pending'`),`session.add_all` + `flush()` 拿 id 后按顺序回填 `prev_chunk_id`/`next_chunk_id`,commit,返回 id 列表(与 chunks 同序)。
  - `vectorize_pending(session, store, embedder) -> int`:查 `vectorize_status=='pending'` 的行 → 拼 `category\nquestions\nanswer` → `embedder.embed_texts`(批)→ `store.upsert([(id, vec)])` → 回填 `vector_id=str(id)`、`vectorize_status='done'`,返回回填数。每块 `assert len(vec)==embedder.dim`。
  - `ingest_document(session, store, embedder, text, content_type, *, max_chars, overlap, source: str | None = None) -> list[int]`:chunk → insert_chunks → vectorize_pending。
  - `migrate_faq(session, store, embedder) -> int`:把 `faq` 表每行转 `Chunk(category=faq.category, questions=faq.question, answer=faq.answer, content_type='faq', ...)` → insert_chunks → vectorize_pending,返回迁移数。
  - CLI `app/jobs/ingest_docs.py`:`python -m app.jobs.ingest_docs` 读 `knowledge/*.md`(逐个 `content_type` 由文件名/约定定)→ ingest_document;再 `migrate_faq`;打印建库块数。

- [ ] **Step 1: 写失败测试 `tests/test_ingest.py`**(fake embedder + 真实 Milvus 临时目录 + SQLite session)

```python
class FakeEmbedder:
    def __init__(self, dim=1024): self.dim = dim
    def embed_texts(self, texts): return [[0.1] * self.dim for _ in texts]

def _chunks():
    return [Chunk(category="售后", questions="退货政策是什么", answer="7天无理由", section_path="退货政策", content_type="faq", is_key_clause=False),
            Chunk(category="售后", questions="运费怎么收", answer="满99包邮", section_path="运费说明", content_type="faq", is_key_clause=False)]

def test_insert_sets_pending_and_pointers(session):
    ids = insert_chunks(session, _chunks())
    assert len(ids) == 2
    a, b = session.get(KnowledgeChunk, ids[0]), session.get(KnowledgeChunk, ids[1])
    assert a.vectorize_status == "pending" and a.next_chunk_id == ids[1] and b.prev_chunk_id == ids[0]

def test_vectorize_pending_backfills(session, tmp_path):
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    ids = insert_chunks(session, _chunks())
    n = vectorize_pending(session, store, FakeEmbedder())
    assert n == 2
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)

def test_interrupt_then_rerun_picks_up_pending(session, tmp_path):  # 验收 #2
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    ids = insert_chunks(session, _chunks())          # 模拟「只写 MySQL,没向量化就挂了」
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "pending" for i in ids)
    assert vectorize_pending(session, store, FakeEmbedder()) == 2   # 重跑补齐
    assert vectorize_pending(session, store, FakeEmbedder()) == 0   # 再跑无事可做(幂等)
    assert all(session.get(KnowledgeChunk, i).vectorize_status == "done" for i in ids)
    hits = store.search([0.1]*1024, top_k=10)
    assert len(hits) == 2

def test_migrate_faq(session, tmp_path):
    session.add(Faq(question="退货政策是什么？", answer="7天无理由", category="售后")); session.commit()
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    assert migrate_faq(session, store, FakeEmbedder()) == 1
    row = session.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "faq").one()
    assert row.questions == "退货政策是什么？" and row.vectorize_status == "done"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 `ingest.py` + CLI**

按 Interfaces。`vectorize_pending` 逐批处理(一批 embed + upsert,块数由 embedder 批大小自然定);`migrate_faq` 里 faq 转 Chunk 复用 `insert_chunks`。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS(4 passed)。

- [ ] **Step 5: Commit**

```bash
git add app/kb/ingest.py app/jobs/__init__.py app/jobs/ingest_docs.py tests/test_ingest.py
git commit -m "feat: dual-write ingest + idempotent reconcile (ch03)"
```

---

### Task 6: 在线检索 `retrieval.py` + `query_faq` 重写

**Files:**
- Create: `app/kb/retrieval.py`
- Modify: `app/tools/db_tools.py`
- Test: `tests/test_retrieval.py`(新建)

**Interfaces:**
- Consumes: `Embedder`(Task 3)、`MilvusStore`(Task 4)、`insert_chunks`/`vectorize_pending`(Task 5)、`KnowledgeChunk`(Task 1)。
- Produces:
  - `retrieve(session, store, embedder, query: str, *, top_k: int = 1, threshold: float = 0.5) -> str | None`:embed(query)→ `store.search(vec, top_k)` → 取最高分,若 `score >= threshold` 回 MySQL 按 id 取 `answer` 返回,否则 `None`。
  - `default_retrieve(query: str) -> str | None`:从 `get_settings()` 建 `SessionLocal` + `MilvusStore` + `Embedder` 调 `retrieve`(集成入口,Task 8 联调用)。
  - `query_faq` 重写为:`ans = default_retrieve(keyword); return ans if ans else "未查到"`,整体包 try/except 降级(Review Focus #5)。

- [ ] **Step 1: 写失败测试 `tests/test_retrieval.py`**(fake embedder 命中/未命中 + 真实 Milvus + SQLite)

```python
class HitEmbedder:
    dim = 1024
    def embed_texts(self, texts): return [[0.1]*self.dim for _ in texts]   # 一切文本同向量 → 恒命中

class MissEmbedder:
    dim = 1024
    def embed_texts(self, texts): return [[-0.1]*self.dim for _ in texts]  # 与存储向量相反 → 恒未命中

def test_retrieve_hit_returns_answer(session, tmp_path):
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    insert_chunks(session, [Chunk(category="售后", questions="运费说明", answer="满99包邮", content_type="policy")])
    vectorize_pending(session, store, HitEmbedder())
    assert retrieve(session, store, HitEmbedder(), "邮费是多少") == "满99包邮"

def test_retrieve_below_threshold_returns_none(session, tmp_path):
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    insert_chunks(session, [Chunk(category="售后", questions="运费说明", answer="满99包邮", content_type="policy")])
    vectorize_pending(session, store, HitEmbedder())
    assert retrieve(session, store, MissEmbedder(), "邮费是多少") is None

def test_retrieve_empty_collection(session, tmp_path):   # Review Focus #4
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    assert retrieve(session, store, HitEmbedder(), "邮费是多少") is None

def test_query_faq_degrades_to_not_found(monkeypatch):   # Review Focus #5
    from app.tools import db_tools
    monkeypatch.setattr(db_tools, "default_retrieve", lambda q: (_ for _ in ()).throw(RuntimeError("milvus down")))
    assert db_tools.query_faq.func("邮费是多少") == "未查到"   # 不抛异常
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 `retrieval.py` + 重写 `query_faq`**

按 Interfaces。`query_faq` 的 `@tool` 装饰器、函数名、参数名 `keyword`、docstring 语义保持不变,只替换函数体。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_retrieval.py tests/test_tools.py -v`
Expected: PASS(新 4 条 + 旧 `test_tools.py` 里对 query_faq 的既有断言不回归;若旧断言按关键词语义写死需同步改为「返回 str」语义,以不破坏契约为准)。

- [ ] **Step 5: Commit**

```bash
git add app/kb/retrieval.py app/tools/db_tools.py tests/test_retrieval.py
git commit -m "feat: vector retrieval replaces keyword query_faq (ch03)"
```

---

### Task 7: 挖知识 `qa_extract.py` + CLI `qa_mining.py`

**Files:**
- Create: `app/kb/qa_extract.py`
- Create: `app/jobs/qa_mining.py`
- Test: `tests/test_qa_extract.py`(新建)

**Interfaces:**
- Consumes: `Embedder`(Task 3)、`MilvusStore`(Task 4)、`insert_chunks`/`vectorize_pending`(Task 5)、`QaExtractionStaging`(Task 1)、`build_extraction_chain` 风格结构化链(复用 `app/services/extraction_service.py` 的 `with_structured_output(method="function_calling")` 模式)。
- Produces:
  - `@dataclass QA`: `question: str`、`answer: str`。
  - `extract_qa(chain, text: str) -> list[QA]`:结构化输出 schema `class QAPairs(BaseModel): pairs: list[QAPair]`(`QAPair: question/answer` 两字段),`chain.ainvoke`/`invoke` 返回 pairs。
  - `normalize(q: str) -> str`:去首尾空白、转小写、去标点(用于去重键)。
  - `dedupe(qas: list[QA]) -> list[QA]`:按 `normalize(question)` 保首弃重。
  - `mine_conversations(session, store, embedder, chain, source_path: str, batch_size: int = 10) -> int`:读 `source_path`(jsonl,每行一个会话文本)→ 按 batch_size 分批 `extract_qa` → 写 `QaExtractionStaging(status='extracted', batch_no=uuid, source_ref=source_path+段)` → 全量 `dedupe`(kept/discarded 写回 status)→ kept 转 `Chunk(content_type='mined', category='历史问答', questions=question, answer=answer)` → insert_chunks + vectorize_pending,返回 kept 入块数。
  - CLI `app/jobs/qa_mining.py`:`python -m app.jobs.qa_mining [--source knowledge/conversations/sample.jsonl] [--batch-size 10]`。

- [ ] **Step 1: 写失败测试 `tests/test_qa_extract.py`**(fake 链 + fake embedder/store + SQLite)

```python
class FakeChain:
    def __init__(self, pairs): self._pairs = pairs
    def invoke(self, inp): return type("R", (), {"pairs": self._pairs})()

def test_extract_qa_returns_pairs():
    chain = FakeChain([type("P", (), {"question": "q1", "answer": "a1"})()])
    out = extract_qa(chain, "任意文本")
    assert out == [QA("q1", "a1")]

def test_dedupe_by_normalized_question():
    qas = [QA("邮费多少？", "a"), QA(" 邮费多少 ", "a"), QA("退货？", "b")]
    assert dedupe(qas) == [QA("邮费多少？", "a"), QA("退货？", "b")]

def test_mine_conversations_end_to_end(session, tmp_path):
    src = tmp_path / "conv.jsonl"
    src.write_text('{"text": "用户:运费怎么收 客服:满99包邮"}\n', encoding="utf-8")
    store = MilvusStore(str(tmp_path/"kv.db")); store.ensure_collection()
    chain = FakeChain([type("P", (), {"question": "运费怎么收", "answer": "满99包邮"})()])
    n = mine_conversations(session, store, FakeEmbedder(), chain, str(src))
    assert n == 1
    assert session.query(QaExtractionStaging).filter(QaExtractionStaging.status == "kept").count() == 1
    row = session.query(KnowledgeChunk).filter(KnowledgeChunk.content_type == "mined").one()
    assert row.vectorize_status == "done" and "满99包邮" in row.answer
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_qa_extract.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 `qa_extract.py` + CLI**

QA 抽取 System Prompt 为纯 Prompt 产物(按工作要求 #1,不以 TDD 钉死,改为 Task 8 用标注样例跑一遍验证抽取质量)。结构化 schema 用 `with_structured_output(QAPairs, method="function_calling")`(ch02 已知 Ollama 上必须显式 `function_calling`)。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_qa_extract.py -v`
Expected: PASS(3 passed)。

- [ ] **Step 5: Commit**

```bash
git add app/kb/qa_extract.py app/jobs/qa_mining.py tests/test_qa_extract.py
git commit -m "feat: QA mining pipeline + dedupe + CLI (ch03)"
```

---

### Task 8: 样例语料 + demo 脚本 + 端到端验收

**Files:**
- Create: `knowledge/return-policy.md`、`knowledge/product-faq.md`、`knowledge/aftersale-manual.md`、`knowledge/conversations/sample.jsonl`
- Create: `scripts/demo_retrieval.py`

**Interfaces:**
- Consumes: 全部上文模块。
- Produces: 样例语料(含「运费说明」,否则验收 #1 无米下锅)+ demo 命令 `python scripts/demo_retrieval.py "邮费是多少"`(内部调 `default_retrieve` 打印命中块与相似度)。

- [ ] **Step 1: 写样例语料**(纯数据,按工作要求 #1 以「跑一遍验证」代替 TDD)

三份 Markdown:`return-policy.md`(含「运费说明:满 99 包邮,不满收 8 元,偏远地区另计」)、`product-faq.md`(真实问法 Q/A)、`aftersale-manual.md`;`sample.jsonl` 含 3+ 段多轮客服对话,至少一段是「运费怎么收 → 满99包邮」类问答。

- [ ] **Step 2: 端到端建库(需 Ollama 起、bge-m3 已拉、MySQL 起)**

```bash
.venv/Scripts/python -m app.jobs.ingest_docs
.venv/Scripts/python -m app.jobs.qa_mining
```
Expected: 打印建库块数;MySQL `knowledge_chunks` 全 `done`、`vector_id` 非空;Milvus `data/milvus.db` 目录生成。

- [ ] **Step 3: 验证验收 #1(换说法召回运费)**

```bash
.venv/Scripts/python scripts/demo_retrieval.py "邮费是多少"
```
Expected: 命中「运费说明」块并打印 `满 99 包邮…`。再跑 `"运费怎么收"`、`"包不包邮"` 均应命中。

- [ ] **Step 4: 验证验收 #2(中断重跑补齐)**

手动:先只 `insert_chunks`(或临时注释 `vectorize_pending`)制造一批 `pending`,再 `python -m app.jobs.ingest_docs` 重跑;Expected: 漏向量化的块被捡起补齐,`pending` 归零、`done` 数 == 块总数。(等价于 `test_interrupt_then_rerun_picks_up_pending`,此处用真实 Ollama+MySQL 再确认一次。)

- [ ] **Step 5: 验证 QA 抽取 Prompt 质量(工作要求 #1)**

用 `sample.jsonl` 跑 `qa_mining`,人工核对抽出的问答对「问题=用户原意、答案=客服原话」无张冠李戴;不符则调 System Prompt 重跑。

- [ ] **Step 6: Commit**

```bash
git add knowledge/ scripts/demo_retrieval.py
git commit -m "feat: sample knowledge corpus + retrieval demo (ch03)"
```

---

## Self-Review 结果

- **Spec 覆盖**:五块功能(文档切分→Task 2/5、挖知识→Task 7、落库结构→Task 1/5、双写→Task 5、在线检索→Task 6)各有一一对应;两条验收(#1→Task 8 Step 3、#2→Task 5 + Task 8 Step 4);「契约不变」→Task 6 Step 3 显式约束。
- **Step 扫描**:每个 Step 落到「跑哪个测试/命令 + 期望输出」,无「TBD/handle edge cases」空话。
- **类型一致**:`Chunk`/`Embedder`/`MilvusStore`/`insert_chunks`/`vectorize_pending`/`retrieve`/`extract_qa`/`dedupe` 等签名在各任务 Interfaces 中首定义、后引用,命名一致。
- **Review Focus**:5 条各落到 Task 2/3/5/6 的测试。
- **比例**:计划与 spec 篇幅相当,测试只写「名字+断言」、代码只写「签名+spec 钉死值」,未转写实现体。
