# 多标签主题分类器(ch10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 微调一个 17 类多标签主题分类器(RoBERTa-wwm-ext 全参微调),旁路批量归类飞轮的低置信度问题,写 `topic_classifications`,喂飞轮后台看主题分布。

**Architecture:** `ml/` 放训练侧(术语表/数据/训练/评测/ONNX 导出),`inference/` 放独立 FastAPI 推理服务(ONNX + onnxruntime),`app/` 放批量 job + 前端接口 + 表。多标签用 RoBERTa 裸编码器 + 手接 `nn.Linear(768,17)` sigmoid 头 + BCE 损失(避开 transformers 5.23 分类头 API 的不确定性)。

**Tech Stack:** transformers 5.23 · torch 2.15(cpu)· sklearn · ModelScope(模型下载,HF 被墙)· ONNX + onnxruntime · FastAPI · SQLAlchemy。

**Spec:** `docs/superpowers/specs/2026-11-21-ch10-topic-classifier-design.md`

## Global Constraints

- **模型**:`hfl/chinese-roberta-wwm-ext`,从 **ModelScope** 下载(HF 被墙),全参微调,不用 LoRA。
- **多标签**:sigmoid + `BCEWithLogitsLoss`;「其他」普通类可共现,全低于阈值 → 归「其他」。
- **transformers 5.23.0 API 已变**(`from_pretrained` 无显式 `num_labels`):用 `AutoModel` 裸编码器 + 手接 `nn.Linear(768,17)`,不依赖分类头 API。
- **训练**:CPU、后台跑、存 checkpoint 断点续训;正则化(dropout/weight decay)+ 验证集 macro-F1 早停。
- **17 类与 3 条硬规则逐字保留**(见 spec §5.1);每类一句边界说明按硬规则补齐。
- **验收「真实问题」** = 手写真实样例 + 池里真实行;人工抽审/错例复核 = 事后复核,非卡点。
- 数据 80/10/10 分层;增强只扩训练集。

## Review Focus

1. **空标签集**:17 个 head 全低于阈值 → 必须归「其他」,不能返回空 labels。→ Task 5 加测试。
2. **多标签同时命中**:「买大了想退」要同时命中「尺码」+「退换货」,不能压成单类。→ Task 5/8 验证。
3. **阈值边界**:单标签 score 恰在阈值上下的判定。→ Task 5 加测试。
4. **推理服务不可用**:批量 job 调不到推理服务时应留痕、不崩主链路。→ Task 7 加测试。
5. **训练中断**:checkpoint 断点续训,别从头重跑。→ Task 4 验证。

---

### Task 1: 术语表 + ML 依赖

**Files:**
- Create: `ml/__init__.py`、`ml/taxonomy.py`
- Modify: `requirements.txt`(加 `modelscope`、`onnx`、`onnxruntime`)
- Test: `tests/test_taxonomy.py`

**Interfaces:**
- Produces:`ml.taxonomy.TOPICS`(17 个 `(name, boundary)` 有序列表)、`topic_names() -> list[str]`、`num_labels = 17`、`OTHER = "其他"`。全仓库唯一权威源。

- [ ] **Step 1: 写失败测试** `tests/test_taxonomy.py`:17 类、名字有序、含 3 条硬规则关键词(退换货含「退」、保修维修含「修」、运费含「钱」、物流含「货」、价保含「差价」、优惠活动含「满减/券」)。
- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_taxonomy.py -v` → FAIL
- [ ] **Step 3: 实现** `taxonomy.py`(17 类 + 每类一句边界,3 条硬规则逐字保留);`pip install modelscope onnx onnxruntime` + 钉进 requirements。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_taxonomy.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch10): 17 类术语表 + ML 依赖"`

---

### Task 2: 模型下载(ModelScope)

**Files:**
- Create: `ml/download.py`
- Test: smoke(下载后确认 `config.json`/`pytorch_model.bin` 存在)

- [ ] **Step 1:** 写 `ml/download.py`:`modelscope` SDK 的 `snapshot_download("hfl/chinese-roberta-wwm-ext", cache_dir="ml/artifacts/model")`(确切 model id 对着 ModelScope 现查)。
- [ ] **Step 2:** 跑一次,确认权重落地;`transformers` 能 `AutoModel.from_pretrained("ml/artifacts/model/hfl/chinese-roberta-wwm-ext")` 加载。
- [ ] **Step 3: 提交** `git commit -m "feat(ch10): ModelScope 下载 RoBERTa-wwm-ext"`

---

### Task 3: 数据管线(清洗 + 分层抽样 + 增强)

**Files:**
- Create: `ml/data/__init__.py`、`ml/data/clean.py`、`ml/data/split.py`、`ml/data/augment.py`、`ml/data/build_dataset.py`
- Test: `tests/test_topic_data.py`

**Interfaces:**
- Produces:`clean(text) -> str`(脱敏手机号/订单号、修格式);`stratified_split(items, seed) -> (train, val, test)`(80/10/10,保每类占比);`augment(text) -> list[str]`(同义词替换/句式微调);`build_dataset() -> None`(产出 `ml/data/{train,val,test}.jsonl`,每行 `{"text","labels"}`)。

- [ ] **Step 1: 写失败测试** `tests/test_topic_data.py`:clean 把 `13812345678` → 掩码、`订单1001` → 掩码;split 三类合计 == 全量且比例 8:1:1;augment 每条至少产出 1 条变体。
- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_topic_data.py -v` → FAIL
- [ ] **Step 3: 实现** clean/split/augment/build_dataset(真实行从 `low_confidence_questions`/`review_queue` 捞,不足的留待 Task 3b 生成)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_topic_data.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch10): 数据管线 清洗+分层抽样+增强"`

---

### Task 3b: 模拟数据生成 + 预标(非 TDD)

**Files:**
- Create: `ml/data/generate_synthetic.py`

- [ ] **Step 1:** 用 qwen3:8b 照术语表 + 边界说明批量造多标签样本到 ~1700(每类 ~100),prompt 强制输出 JSON `{"text","labels"}`;产 `ml/data/raw_synthetic.jsonl`。
- [ ] **Step 2:** 抽 30 条人工核:标签是否贴合、多标签是否到位;不合就改 prompt 重生成。
- [ ] **Step 3:** 合并真实 + 模拟 → 跑 `build_dataset` 出 train/val/test。
- [ ] **Step 4: 提交** `git commit -m "feat(ch10): 大模型生成 1700 条多标签模拟样本"`

---

### Task 4: 训练(非 TDD,训练类)

**Files:**
- Create: `ml/train.py`

**Interfaces:**
- Produces:`train() -> None`:加载 RoBERTa 裸编码器 + `nn.Linear(768,17)` 头,BCE 损失,AdamW(lr 2e-5, wd 0.01),dropout 0.1,验证集 macro-F1 早停,存 `ml/artifacts/best.pt` + 每 N 步 checkpoint;训练曲线打 `log/train_ch10.log`。

- [ ] **Step 1:** 写训练循环(手写,不依赖 Trainer 的分类头 API);先在小样本上冒烟(10 条能跑通 forward/backward)。
- [ ] **Step 2:** 后台全量训练(train/val 数据),盯验证集 F1 与早停。
- [ ] **Step 3:** 确认 best.pt 落地 + 训练日志可读。
- [ ] **Step 4: 提交** `git commit -m "feat(ch10): RoBERTa-wwm-ext 全参微调 + 早停"`

---

### Task 5: 评测(每类 P/R/F1 + 混淆矩阵 + 推理函数)

**Files:**
- Create: `ml/eval.py`、`ml/predict.py`
- Test: `tests/test_topic_eval.py`

**Interfaces:**
- Produces:`predict(model, tokenizer, texts, threshold=0.5) -> list[list[str]]`(sigmoid 后按阈值取标签,空则归「其他」);`evaluate(model, tokenizer, test_items) -> dict`(每类 precision/recall/f1 + macro/micro + 每类 2×2 混淆矩阵),写 `ml/artifacts/eval_report.json` + 错例清单 `ml/artifacts/errors.jsonl`。

- [ ] **Step 1: 写失败测试** `tests/test_topic_eval.py`:用假分数/假 labels 测 `predict`(阈值边界、空标签归「其他」、多标签同中)与 `evaluate` 的 per-class F1 数值。
- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_topic_eval.py -v` → FAIL
- [ ] **Step 3: 实现** predict/evaluate(复用 sklearn 的 precision_recall_fscore_support)。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_topic_eval.py -v` → PASS;对 best.pt 跑测试集,产出 eval_report.json。
- [ ] **Step 5: 提交** `git commit -m "feat(ch10): 评测 每类F1+混淆矩阵 + 推理函数"`

---

### Task 6: ONNX 导出 + 独立推理服务

**Files:**
- Create: `ml/export_onnx.py`、`inference/app.py`
- Test: `tests/test_inference.py`(smoke)

**Interfaces:**
- Produces:`export_onnx() -> None`(torch.onnx 导出 `ml/artifacts/topic_classifier.onnx`,动态 batch);`inference/app.py`(FastAPI,端口 8001):`POST /classify`(body `{"texts":[str]}` → `{"labels":[[str]]}`)+ `/health`。

- [ ] **Step 1:** 导出 ONNX;写 `inference/app.py`(onnxruntime 加载,输入 ids/attention_mask,输出 sigmoid → 阈值 → 标签)。
- [ ] **Step 2:** 起服务,`curl /health` 200;`POST /classify` 用「买大了想退」能同时返回「尺码」「退换货」。
- [ ] **Step 3: 提交** `git commit -m "feat(ch10): ONNX 导出 + 独立推理服务"`

---

### Task 7: topic_classifications 表 + 批量 job

**Files:**
- Modify: `app/db/models.py`(加 `TopicClassification`)、`db/schema.sql`(DDL)、`app/jobs/classify_topics.py`、`app/core/jobs.py`(注册)
- Test: `tests/test_topic_job.py`

**Interfaces:**
- Produces:`TopicClassification(id, low_conf_id FK, labels JSON, created_at)`;`classify_topics(batch_size=32) -> dict`:攒未归类的 `low_confidence_questions` → 调推理服务 `/classify` → 写行(服务不可用则留痕跳过,不崩)。

- [ ] **Step 1: 写失败测试** `tests/test_topic_job.py`:monkeypatch 推理服务返回固定 labels → classify_topics 写 N 行、labels 正确;monkeypatch 服务抛异常 → 不崩、返回 processed=0。
- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_topic_job.py -v` → FAIL
- [ ] **Step 3: 实现** model + DDL + job + 注册 `classify-topics`。
- [ ] **Step 4: 跑测试确认通过** `pytest tests/test_topic_job.py -v` → PASS
- [ ] **Step 5: 提交** `git commit -m "feat(ch10): topic_classifications 表 + 批量归类 job"`

---

### Task 8: 前端页面(Vibe Coding)

**Files:**
- Create: `app/api/routes/topics.py`、`app/api/routes/acceptance.py`、`static/topics.html`/`topics.js`、`static/acceptance*.html`
- Test: `tests/test_static.py`(补页面断言)

- [ ] **Step 1:** 后端聚合:`/api/topics/distribution`(各类目计数)、`/api/topics/questions?label=`、`/api/acceptance/*`(读 eval_report.json / errors.jsonl / 数据产物)。
- [ ] **Step 2:** 页面:主题分布(柱状图,纯 CSS/JS)、类目问题列表、验收总览/评测详情/数据产物/错例复核;导航接进统一 topnav。
- [ ] **Step 3:** `node --check`;`pytest tests/test_static.py` 通过。
- [ ] **Step 4: 提交** `git commit -m "feat(ch10): 主题分布页 + 验收页"`

---

### Task 9: 手写真实样例验收 + 全量回归

**Files:**
- Create: `ml/data/hand_written.jsonl`

- [ ] **Step 1:** 写一批贴近真实的多诉求样例(「买大了想退」「我要开发票」「快递还没到」「能便宜点吗」等 ~30 条,人工标多标签)。
- [ ] **Step 2:** 用 best.pt / 推理服务跑这批 + 池里真实行,人工核命中情况(验收 2、3)。
- [ ] **Step 3:** 全量 `pytest` 绿;功能演示命令写进 dev-notes finish。
- [ ] **Step 4: 提交** `git commit -m "chore(ch10): 手写真实样例验收 + 全量回归"`
