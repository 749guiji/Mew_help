# ch03 · RAG 知识库 — 设计文档

日期:2026-11-17
状态:待评审(草案 → builder check → 用户审批)

## 1. 意图与为什么

客服系统现有的 `query_faq` 用 `Faq.question LIKE '%关键词%'` 做子串匹配,只能命中字面相同的问题。用户换一种说法(「邮费是多少」「运费怎么收」「包不包邮」)就漏召回,ch02 验收时已暴露「邮费 = 未查到」。本章把 `query_faq` 的内部实现从「关键词查表」升级为「BGE-M3 向量语义检索 + Milvus 召回」,让换说法的问法也能召回同一条知识。工具对外契约(`query_faq(keyword: str) -> str`)保持不变,只换内核。

同时把「知识」本身工程化:离线从两类原料建库——(a) 结构化 Markdown 文档(退货政策、商品 FAQ、售后手册),(b) 历史客服对话里 LLM 抽出的问答对。落库采用 MySQL(原文权威源)+ Milvus(向量)双写,保证可溯源、可幂等重跑。

## 2. 目标 / 非目标 / 反目标

**目标**
1. 离线文档管道:Markdown 按标题层级做结构感知切分;超长递归切;块间重叠且裁到最近句号;大表格按行切且每块复制表头。
2. 离线挖知识管道:历史客服对话分批喂 LLM 抽问答对,先进暂存表,再整体去重入库。
3. 落库结构:每条知识带 `category` / `questions` / `answer`,三者拼成一段文本做向量化;`section_path` / `content_type` / `is_key_clause` / 前后块指针是元数据,只存不进向量。
4. 双写落库:先 MySQL 记「待向量化」,再 Milvus 拿主键回填、转「已向量化」;按主键幂等,中断后可重跑补齐。
5. 在线检索:问题向量化后到 Milvus 按相似度取 Top-K,替换 `query_faq` 关键词实现。

**非目标(本章不做)**
- 关键词召回、混合检索、重排;本章只跑 dense 向量单路。
- Milvus 集群 / 生产化部署;本章用 Milvus Lite 单机嵌入式。
- embedding 微调;直接用预训练 BGE-M3。

**反目标(做了反而算失败)**
- 改 `query_faq` 的对外签名或返回类型,导致 ch02 工具链和调用方回归。
- 引入重型 RAG 框架(LangChain VectorStore/Retriever/RetrievalQA),让自定义切分规则被框架约束。
- 让离线建库和在线服务同时打开同一个 Milvus Lite 库(Milvus Lite 单进程约束),导致文件锁冲突。

## 3. 约束

- 技术栈定死:BGE-M3(嵌入)、Milvus(向量)、MySQL(原文权威源)、LangChain(仅 LLM/嵌入调用)、FastAPI(在线)。实现中发现矛盾或走不通,停下问用户,不自行换方案。
- 嵌入模型 BGE-M3 复用现有 Ollama(`base_url=http://localhost:11434`,当前 Ollama v0.33.3),维度 1024。需在计划阶段查证 Ollama 嵌入端点是 OpenAI 兼容 `/v1/embeddings` 还是原生 `/api/embed`(不凭记忆)。
- Milvus 用 **Milvus Lite 3.x**(纯 Python,Windows 可跑;2.x 不支持 Windows)。Windows 上 `pip` 会跳过 `pymilvus[milvus-lite]` 的 extra,**必须显式 `pip install milvus-lite`**。`MilvusClient("./milvus.db")` 生成的是目录不是单文件,需 gitignore。单 data_dir 单进程。
- 操作系统 Windows 11;现有依赖 Python 3.14 / pymysql / sqlalchemy 2.x / langchain 1.x。新增依赖需验证在 Python 3.14 下可装(ch01 已确立「装真实依赖 introspect 签名」的核实方法)。
- 工具调用走 ch02 既有 `ToolExecutor`,`query_faq` 仍是 `@tool`,仅替换函数体。

## 4. 方案(Approaches)

### 4.1 Milvus 部署形态
- **A. standalone 服务器**(docker-compose 加 etcd + minio + milvus):生产形态、多进程可并发访问;但要起 3 个容器、多一堆运维面,且 demo 阶段重。
- **B. Milvus Lite 嵌入式**(`MilvusClient("./milvus.db")`):零基础设施、纯 Python、同 pymilvus API,后续换 URI 即可迁到真服务器;代价是单进程、开发/测试量级。
- **选 B(用户拍板)**:契合「先离线建库、再在线检索」的本章节奏,迁移路径清晰(只改 URI)。

### 4.2 切分实现
- **A. 手写结构感知切分器**:解析 Markdown 标题树 → 按 section 切 → 超长按段落/句子递归 → 重叠裁句号 → 表格按行切复制表头。自定义规则直接、可单测。
- **B. LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`**:覆盖「按标题分 + 递归分」,但「重叠裁到最近句号」「表格复制表头」不在其能力内,还得自己补,反而夹生。
- **选 A**:本章切分规则本身就是自定义的,手写比套框架干净。

### 4.3 挖知识「定时任务」形态
- **A. CLI 命令**(`python -m app.jobs.qa_mining`):无新依赖、可被 cron/计划任务调度、重跑即幂等、测试直接调函数。
- **B. APScheduler 进程内定时**:真正定时,但引入新依赖、进程内状态、与单进程 Milvus Lite 约束冲突(服务常驻会一直占着库)。
- **选 A(用户确认「搞个命令启动,后面随时部署到定时服务」)**。

### 4.4 其余「怎么建」已定
- 向量检索用官方 `pymilvus` 直写,不套 LangChain 的 VectorStore 抽象。
- 嵌入调用复用 LangChain(优先 `langchain-openai` 的 `OpenAIEmbeddings` 指向 Ollama,若端点不兼容则 `langchain-ollama` 的 `OllamaEmbeddings` 或直连 `/api/embed`),计划阶段查证后定。

## 5. 架构与数据流

```
[离线·文档管道]
knowledge/*.md ──chunker──▶ KnowledgeChunk(pending) ──embed──▶ Milvus.upsert ──▶ 回填 vector_id + done

[离线·faq 迁移]
faq 表(5 条种子) ──转──▶ KnowledgeChunk(content_type='faq', pending) ──▶ 同上向量化出口

[离线·挖知识管道]
历史对话 ──分批 LLM 抽 QA──▶ qa_extraction_staging(extracted) ──整体去重──▶ kept 转 KnowledgeChunk(pending) ──▶ 同上向量化出口

[在线]
用户问题 ──query_faq──▶ embed(question) ──▶ Milvus.search Top-K ──▶ 按 id 回 MySQL 取 answer ──▶ 返回
```

**向量化文本** = `category + "\n" + questions + "\n" + answer`(三者拼一段)。
**Milvus 集合 `knowledge`** 只存 `id`(int64 主键)+ `vector`(float_vector, dim 1024)。原文与元数据一律在 MySQL `knowledge_chunks` 里,检索时拿 Milvus 返回的 id 回 MySQL 取 answer。
**主键对齐**:Milvus 集合主键 = MySQL `knowledge_chunks.id`(int64,1:1)。`vector_id` 列存 `str(id)`,作为「已写入 Milvus」的回填凭证(NULL = 未写入)。

## 6. 关键设计决策

1. **双写幂等**:INSERT MySQL(`vectorize_status='pending'`)→ 取 `id` → embed → `MilvusClient.upsert([{id, vector}])`(upsert 按主键幂等)→ UPDATE MySQL `vector_id=str(id)`, `vectorize_status='done'`。重跑 = 扫 `vectorize_status='pending'` 的行重新 embed + upsert + 回填,天然补齐漏向量化的块(验收 #2)。
2. **切分规则**:按标题层级建章节路径;section 正文 ≤ `MAX_CHUNK` 则整块,否则按段落→句子递归切,块间 `OVERLAP` 重叠,重叠边界裁到最近 `。!?！？`;Markdown 表格按行切,每块前缀表头行 + 表上方标题/说明。
3. **questions 的填法**:`content_type='faq'`(商品 FAQ)和挖出的问答对,`questions` 填真实问法;`content_type='policy'/'manual'` 这类没有天然问题的,`questions` 填所在章节标题、`category` 填上级标题路径。
4. **prev/next 指针**:chunk 的 id 是自增,前后块指针在「同文档所有 chunk 落库后」二次回填(无法在插入时就知道 next 的 id)。
5. **挖知识去重**:「整体去重」= 抽完后对 `qa_extraction_staging` 做一批归一化(去空白/标点/大小写)去重,保留一条 `kept`、其余 `discarded`;归一化去重为基线,embedding 近义去重列为可选增强(本章不强制)。
6. **单进程约束下的运行顺序**:demo 与验收按「先跑 CLI 建库(进程退出释放库)→ 再起 FastAPI 服务做在线检索」的顺序。CLI 与服务不同时开同一个 `.db` 目录。
7. **`query_faq` 契约**:函数签名 `query_faq(keyword: str) -> str`、`@tool` 名、docstring 语义不变;内部改为向量检索,`keyword` 语义上变成「自然语言问题」。Top-K 取 K=1 时返回该块 `answer`(沿用「查不到返回『未查到』」的兜底语义,改为「相似度低于阈值则未查到」)。
8. **Milvus metric_type = COSINE**:BGE-M3 输出归一化,COSINE 与 IP 等价,是安全默认,建集合时定死(改它要重建集合)。相似度阈值 `retrieval_threshold` 做成配置项,默认 0.5;验收阶段用样例语料标定一个能把「命中/未命中」分开的值再落定。
9. **`retrieval_top_k` 语义**:搜索广度 = 从 Milvus 取 Top-K 候选;`query_faq` 固定返回 **Top-1 候选的 `answer`**(score ≥ 阈值,否则「未查到」)。本章 `retrieval_top_k` 默认 1,更高 K 留给未来混合检索/重排,不在本章使用。
10. **存量 `faq` 表迁移**:ch03 后 `query_faq` 不再读 `faq`。为不让 5 条种子答案失联,把 `faq` 作为**第三种入料源**迁进 `knowledge_chunks`(`content_type='faq'`,questions=question、answer=answer、category=category),与 Markdown、挖知识共用同一向量化出口。`faq` 表本身保留当「原始录入」,不删。
11. **挖知识输入源**:本章 loader 读样例对话文件(`knowledge/conversations/*.jsonl`),`source_ref` 记「文件 + 段」;不读 ch02 的 `conversations`/`messages` 表(那里是模型自己生成的回复,挖它属循环)。loader 输入做成可配置路径,后续接真实客服日志或 DB 只换 source。
12. **`is_key_clause`**:本章纯元数据(只存不进向量、不参与召回过滤/加权),默认 0。是否标 1 留给 builder 定一个简单启发式(如命中「必须/不得/一律/免责/责任」等词),本章不参与检索。

## 7. 模块划分(新增)

```
app/kb/__init__.py
app/kb/chunker.py        # 结构感知切分(纯函数,可单测)
app/kb/embedding.py      # BGE-M3 嵌入客户端(Ollama)
app/kb/milvus_store.py   # Milvus Lite:建集合 / upsert / search
app/kb/ingest.py         # 文档管道 + faq 迁移 + 双写 + 幂等重跑
app/kb/retrieval.py      # 在线:embed → search → 回 MySQL 取 answer
app/kb/qa_extract.py     # LLM 抽 QA(结构化输出)+ 归一化去重
app/jobs/__init__.py
app/jobs/ingest_docs.py  # CLI:python -m app.jobs.ingest_docs
app/jobs/qa_mining.py    # CLI:python -m app.jobs.qa_mining [--dedupe]
```

改动:
- `app/db/models.py` 加 `KnowledgeChunk`、`QaExtractionStaging` 两个 ORM(对齐用户给的 DDL)。
- `db/schema.sql` 加 `knowledge_chunks`、`qa_extraction_staging` 建表 DDL(用户已给,微调章号注释为 ch03)。
- `app/tools/db_tools.py` 的 `query_faq` 改调 `app/kb/retrieval`。
- `app/core/config.py` 加 `milvus_uri` / `milvus_collection` / `embedding_model` / `embedding_dim` / `retrieval_top_k` / `chunk_max_chars` / `chunk_overlap_chars`。
- `requirements.txt` 加 `pymilvus==3.x`、`milvus-lite==3.x`(Windows 显式装),及嵌入客户端依赖。
- `.gitignore` 加 Milvus Lite 数据目录。

## 8. 数据模型

用户已提供 DDL,本章照用,落盘到 `db/schema.sql`(实现 Task 1)。字段类型已钉死,无歧义:`questions` 是 **TEXT、多个问法换行分隔**(非 JSON 数组);两个状态 ENUM 都是**英文值**(`vectorize_status` 取 `pending`/`done`,`status` 取 `extracted`/`kept`/`discarded`),沿用 ch02 已修的 `SET NAMES utf8mb4` 防 mojibake。原文如下:

```sql
-- ch03 · RAG 基础 · 建表 DDL
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```

## 9. 样例语料设计(本章自建,后续可换真实语料)

新增 `knowledge/` 目录(仓库内,gitignore 不含,版本可控):

1. `knowledge/return-policy.md`(退货政策,`content_type=policy`):含「七天无理由退货」「**运费说明**(满 99 包邮,不满收 8 元,偏远另计)」「质量问题退货」「退货流程」等章节。
2. `knowledge/product-faq.md`(商品 FAQ,`content_type=faq`):若干真实问法 Q/A 对。
3. `knowledge/aftersale-manual.md`(售后手册,`content_type=manual`):退款到账时间、发票开具、换货规则等章节。
4. `knowledge/conversations/sample.jsonl`(历史客服对话):多段用户-客服多轮对话,其中含「运费怎么收」类问答,供挖知识管道演示抽取。

「运费说明」必须落在样例语料里,否则验收 #1(「邮费是多少」答对)无米下锅。

## 10. 验收标准映射

1. **「邮费是多少」能召回运费说明并答对**:在线检索测试 + demo 命令。`embed("邮费是多少")` → Milvus Top-1 命中运费说明块 → 返回其 answer。
2. **故意中断建库任务再重跑,漏向量化的块被补齐**:幂等测试。造若干 `pending` 块、模拟部分已 upsert,重跑 reconcile,断言全部转 `done` 且 Milvus 里数量对齐。

## 11. 已知风险与限制

- **Milvus Lite 单进程**:CLI 与 FastAPI 不能并发开同一库;后续定时化需换真 Milvus 服务器(改 URI 即可)。
- **Milvus Lite 3.x 无浮点压缩/无 PQ/无鉴权**:开发量级够用,生产不适用(已记入非目标)。
- **Ollama 嵌入端点形态未定**:`/v1/embeddings` vs `/api/embed`,计划阶段查证(不凭记忆),决定嵌入客户端实现。
- **Python 3.14 + 新依赖兼容性**:`pymilvus`/`milvus-lite` 及嵌入库需在 3.14 下可装;计划阶段用真实依赖 introspect 验证。
- **去重粒度**:归一化去重只拦字面重复,近义重复(「邮费多少」「运费多少钱」)需要 embedding 近义去重(可选增强,本章不强制)。

## 12. 留给 builder 的

- 各模块的具体函数签名、异常处理、日志与可观测性(保持与 ch02 `ToolExecutor` 一致的风格)。
- 切分参数(`MAX_CHUNK`/`OVERLAP`)默认值及是否纳入配置。
- 表格检测的具体正则/解析策略(是解析 Markdown pipe 表还是自定义标记)。
- 测试组织与 fake 策略(嵌入客户端 fake、Milvus 用真实 Lite 临时目录或 fake、LLM 抽 QA 用 fake structured output)。
- 去重是否纳入 embedding 近义去重的实现取舍(基线只做归一化)。
- demo 脚本(`scripts/demo_retrieval.py` 或等价)的具体形态。
