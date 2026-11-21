# 第十章「多标签主题分类器」— 设计文档

日期:2026-11-21
状态:待 builder check / 待用户批准

## 1. 意图与动机

飞轮(ch09)把答不上的低置信度问题攒进 `low_confidence_questions` 池,但还没有「按主题批量归类」的能力——不知道哪类问题最多、该先补哪块知识。本章微调一个自己的 **17 类多标签主题分类器**(RoBERTa-wwm-ext 全参微调),挂在旁路批量归类,结果写进 `topic_classifications` 表,喂给飞轮后台看主题分布。实时对话主链路不调它。

## 2. 目标 / 非目标 / 反目标

**目标**
1. 数据四步:清洗(脱敏/修错别字格式)、归并术语表(17 类权威类目 + 边界说明)、分层抽样(80/10/10)、增强(同义词替换/句式微调,只扩训练集)。
2. 标注:多标签(一句提几个诉求打几个标签),大模型照术语表预标 + 人工抽审。
3. 训练:RoBERTa-wwm-ext 全参微调(不用 LoRA),验证集盯分数、正则化 + 早停防过拟合。
4. 评测:测试集按类目算 P/R/F1 + 混淆矩阵,外加人工抽判错样本。
5. 部署:ONNX + 独立 FastAPI 推理服务,旁路批量归类写 `topic_classifications`;主对话链路不碰。
6. 前端:飞轮后台主题分布页 + 验收总览/评测详情/数据产物/错例复核/类目问题列表。

**非目标**
- LoRA / QLoRA(大模型才用)。
- embedding 微调、意图识别微调。

**反目标(跑通也算失败)**
- 训练数据里大量造假标签、或标签和术语表边界打架。
- 用 HF 下载模型(被墙,直接走不通)。
- 主对话链路引入分类器延迟。
- 把「多标签」做成「多分类(softmax)」——一句多诉求会被压成一个类。

## 3. 约束

- 技术选型定死:哈工大讯飞 RoBERTa-wwm-ext、全参微调、ONNX + 独立 FastAPI 推理服务。不换。
- 环境硬约束(侦查已确认):本机 CPU-only(`torch 2.15.0+cpu`);HuggingFace 被墙(000)但 ModelScope 通(302)→ 模型从 ModelScope 拉同款 `hfl/chinese-roberta-wwm-ext`;`transformers==5.23.0`(大版本,API 对着本机装的版本核对)。
- 飞轮池当前近乎空 → 1700 条基础量几乎全由大模型照术语表造模拟数据补足(用户已认可)。
- 17 类与边界按用户给定逐字保留(见 §5.1)。

## 3.1 评审决议(builder check,用户已拍板)

- **验收「真实问题」** = 手写一批贴近真实的多诉求样例(如「买大了想退」「我要开发票」「快递还没到」)+ 池里能捞到的真实行,跑归类当验收 #2。
- **人工抽审/错例复核 = 事后复核,非卡点**:模型先按大模型预标标签训练完、出报告,人工再在抽审页/错例页复核、决定要不要补数据。
- **分类单位 = raw 问题**:`topic_classifications` 关联 `low_confidence_questions.id`,分布页按 raw 计数(去重加权后续可加)。
- **「其他」语义**:普通类、可与其他类共现;17 个 head 全部低于阈值 → 归「其他」(保证每句至少一个标签)。
- **训练**:后台跑 + 存 checkpoint 断点续训(防中途被系统回收)。
- **边界说明**:用户只给了 17 类名 + 3 条硬规则,每类一句边界说明由我按硬规则补齐(硬规则逐字保留)。

## 4. 方案取舍(approaches)

### 4.1 模型下载
- **A(选)ModelScope 拉同款权重**。HF 被墙走不通,ModelScope 镜像 `hfl/chinese-roberta-wwm-ext` 同一份权重,`transformers` 直接加载。**选**。
- B:硬试 HF。已确认 000,死路。**弃**。

### 4.2 分类头
- **A(选)多标签 sigmoid + BCE**。RoBERTa [CLS] → dropout → Linear(768,17) → sigmoid,`BCEWithLogitsLoss`。一句多诉求能同时命中多个类。**选**。
- B:softmax 多分类。一句话只能落一个类,违背「多标签」需求。**弃**。

### 4.3 训练数据来源
- **A(选)真实优先 + 大模型造模拟补足**。真实从 `low_confidence_questions.raw_question` / `review_queue.normalized_question` 捞(当前近乎空);不足的照术语表 + 边界说明让 qwen3:8b 批量造多标签样本到 ~1700(每类 ~100)。**选**(用户已认可)。
- B:纯真实。池子空,做不成。**弃**。

### 4.4 推理部署
- **A(选)ONNX + 独立 FastAPI 服务**(单独端口,如 8001)。主对话链路不调;旁路批量 job 调 `/classify`。**选**(用户已定)。
- B:把模型进程内加载进主 FastAPI。CPU 上吃内存、拖慢主服务。**弃**。

## 5. 已定设计(how)

### 5.1 术语表(单一权威源 `ml/taxonomy.py`)

17 类(顺序固定),每类一条边界说明;近邻靠边界划开:

退换货(退/换), 物流(管货), 尺码, 发票, 质量问题, 运费(管钱), 优惠活动(券/满减), 价保(补差价), 支付, 订单修改, 库存补货, 商品信息, 保修维修(修), 账号, 会员积分, 评价, 其他。

边界硬规则(用户给定):「修」归保修维修、「退」归退换货;运费管钱、物流管货;价保是补差价、优惠活动是券和满减。所有下游(生成/预标/训练/推理/前端)都引用这一份,不复制。

### 5.2 数据管线 `ml/data/`

- `build_dataset.py`:捞真实问题 → 清洗(脱敏:手机号/订单号掩码;修格式) → 不够则 `generate_synthetic.py` 用 qwen3:8b 照术语表造多标签样本到 1700 → LLM 预标 → 分层抽样 80/10/10(多标签用 iterative stratification 保每类占比) → `augment.py` 只扩训练集(同义词替换 + 句式微调)。
- 产物:`ml/data/{train,val,test}.jsonl`,每行 `{"text": "...", "labels": ["退换货", "尺码"]}`。

### 5.3 训练 `ml/train.py`

- 加载 `hfl/chinese-roberta-wwm-ext`(本地 ModelScope 下载的权重),换 17 维 sigmoid 头。
- `AdamW(lr=2e-5)`,weight decay 0.01,dropout 0.1,`BCEWithLogitsLoss`;按验证集 macro-F1 早停,存 best checkpoint + 训练曲线日志。
- CPU 上跑,分 batch 控制内存;每步把 train/val 损失 + F1 打到 `log/` 便于后台盯。

### 5.4 评测 `ml/eval.py`

- 测试集上按类目算 precision / recall / F1(多标签→每类二分类),加 macro/micro 平均;每类打 2×2 混淆矩阵。
- 输出 `ml/reports/eval_report.json`(喂前端验收页)+ 错例清单(`predicted != gold` 的样本)供人工复核。

### 5.5 部署

- `ml/export_onnx.py`:`torch.onnx` 导出(动态 batch),产出 `ml/artifacts/topic_classifier.onnx`。
- `inference/`(独立 FastAPI,端口 8001):加载 ONNX(`onnxruntime`),`POST /classify`(body `{"texts": [...]}` → `{"labels": [[...], ...]}`)+ `/health`。
- `topic_classifications` 表(新):`id` / `low_conf_id`(FK `low_confidence_questions.id`)/ `labels`(JSON 多标签)/ `created_at`。
- 旁路批量 job `app/jobs/classify_topics.py`:攒够一批未归类的 `low_confidence_questions` → 调推理服务 → 写 `topic_classifications`;注册进 `JOB_REGISTRY`(手动触发)。

### 5.6 前端(Vibe Coding)

- `/topics`:主题分布页(各类目问题量,简单图表,一眼看出哪类堆得多)。
- `/topics/questions?label=`:类目问题列表。
- `/acceptance`:验收总览;`/acceptance/eval` 评测详情;`/acceptance/data` 数据产物;`/acceptance/errors` 错例复核。
- 后端聚合接口:`/api/topics/distribution`、`/api/topics/questions`、`/api/acceptance/*`(读 `eval_report.json` + `topic_classifications` + 数据产物)。

## 6. 留给 builder 的细节

- ModelScope 上 `hfl/chinese-roberta-wwm-ext` 的确切 model id 与下载方式(`modelscope` SDK `snapshot_download` 或 git)。
- `transformers==5.23.0` 里 `AutoModelForSequenceClassification` / tokenizer / `Trainer`(或手写训练循环)的确切 API——对着本机装的版本核对,不凭记忆。
- 17 类 synthetic 生成的 prompt 与「每类 ~100 条多标签样本」的分布控制。
- 阈值:默认 0.5,验证集上按 F1 微调 per-label 阈值(可选)。
- ONNX 导出 + `onnxruntime` 推理的输入输出名。

## 7. 验收映射

1. 测试集各类目 F1 + 混淆矩阵报告跑出来 → §5.4,产出 `eval_report.json` + 前端 `/acceptance/eval`。
2. 一批真实问题跑归类,后台主题分布页能看到统计 → §5.5/5.6。
3. 「买大了想退」能同时命中多类 → §5.2 多标签标注 + §4.2 sigmoid 头。
