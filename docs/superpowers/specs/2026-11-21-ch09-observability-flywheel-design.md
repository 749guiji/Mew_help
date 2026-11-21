# 第九章「可观测性与数据飞轮」— 设计文档

日期:2026-11-21
状态:待 builder check / 待用户批准

## 1. 意图与动机

给客服系统装「眼睛」+ 把「答不上」变成「改进燃料」。

「眼睛」= Langfuse 全链路 trace:每条请求的每个节点(prompt / 工具调用 / 检索结果 / token 消耗 / 耗时)都能在界面上铺开看,并能在意图维度统计 token 花销。

「燃料」= 数据飞轮:三个入口(检索置信度低 / 生成自评不足 / 用户点 👎)把问题送进 ch04 的 `low_confidence_questions` 池并带当轮召回片段快照;再标准化→查重→进 `review_queue` 待审;人工审核通过走 ch03 落库流程写回知识库。这样「答不上」的问题最终变成知识库里的新知识点,下次能答上。

## 2. 目标 / 非目标 / 反目标

**目标**
1. Langfuse:设 env、挂一次回调,完整 trace 树可见;意图识别结果进 trace 元数据,可按意图统计 token 花销。
2. Cost Control:按意图维度聚合 token 消耗,看出钱花在哪类问题。
3. 三入口落池(各自标 source + 带召回快照):检索置信度低(升级 evidence_confidence 闸)/ 生成自评不足(接进图)/ 用户 👎(接后端)。
4. 飞轮流水线:标准化→查重→入 `review_queue`(命中累加 occurrence_count、不新建行;归并落点记 `matched_review_id`)。
5. 评估流水线:复用 ch04 评估集与指标,每轮落 `eval_runs`,看趋势。
6. 前端:待审队列后台页(列表/详情/通过/驳回,通过走 ch03 落库)。

**非目标**
- 低置信度问题按主题归类的微调分类器(下一步)。
- 自动通过审核(人工把关)。

**反目标(跑通也算失败)**
- 无 Langfuse 环境时崩掉整条链路(未配置就静默跳过)。
- 拍脑袋定 evidence_confidence 阈值(必须用评估集校准)。
- 查重命中却新建重复行(必须累加 + 记归并落点)。
- 👎 落池时把「没走检索」的轮次硬编造出召回快照(该空就空)。

## 3. 约束

- 技术选型定死:Langfuse(开源自部署,链路数据不出本机)。不换。
- 闸位置不动:仍卡在知识类检索之后、进 Agent 之前。
- 涉及 Langfuse / LangGraph callback / FastAPI / SQLAlchemy 的 API,动手前 Context7 核实(本环境被墙则 WebSearch 兜底)。
- `review_queue` / `eval_runs` DDL 与 `low_confidence_questions` 加列按用户给定逐字保留(见 §5.1)。

## 4. 方案取舍(approaches)

### 4.1 Langfuse 回调挂载
- **A(选)每次 invoke 的 config 里挂 `callbacks=[handler]` + metadata**。LangGraph 的 `compile()` 不接收 callback,callback 走 `config` 传播到所有节点,等价于「挂一次」。handler 惰性建(env 未配则 None,静默跳过)。**选**。
- B:给每个节点 `@observe` 装饰。侵入图节点、与 LangChain 回调重复记两层。**弃**。

### 4.2 意图 → trace 元数据(动态值)
- **A(选)事后回写 trace tags**:意图在 graph 内产出,而 config 的 metadata 是 invoke 前定的,先只挂 user/session;图跑完后用本次 handler 的 trace id 把 `intent:{意图}` 写回 trace tags,Langfuse 仪表盘按 tag 聚合 token 花销。**选**。
- B:意图节点里直接调 Langfuse 更新 trace。把可观测性耦合进图节点,图不应知道 Langfuse。**弃**。

### 4.3 evidence_confidence 信号与阈值
- **A(选)三信号加权 + 评估集校准**:`evidence_confidence = f(精排Top1分, Top1−Top2分差, 有效证据条数)`。权重与阈值用 ch04 评估集(命中/未命中)实际跑一遍扫出来,选分离度最好的组合,不拍脑袋。**选**。
- B:单一 Top1 分阈值。丢掉了「证据是否唯一且突出」这个对拒答至关重要的信号。**弃**。

### 4.4 self_check 落点与语义
- **A(选)独立 `self_check` 节点**,知识类路由 `agent → self_check → log`。与闸对称、可单测。**选**(用户已拍板)。
- B:塞进 agent 节点内部。难测、混在 ReAct 循环里。**弃**。
- **语义(用户拍板 A):事后判断 + 落池,不拦答案**。ch08 起答案在 agent 节点里边生成边流式发给用户,self_check 放在 agent 之后无法「拒答」已流出去的答案;故 self_check 只判「答案够不够证据支撑」,不够就落池(供飞轮),**不改用户已收到的答案、不追加拒答话术**。保留流式、零额外延迟。备选「前置闸」会退掉知识类流式,弃。

### 4.5 飞轮触发
- **A(选)手动 job**(`/api/jobs/flywheel` + CLI):审核是人工节奏,手动更可控。**选**(用户已拍板)。
- B:定时任务。无端引入调度依赖。**弃**。

## 5. 已定设计(how)

### 5.1 数据模型(按用户给的 DDL)

用户消息里给了完整 `CREATE TABLE` DDL(`review_queue` / `eval_runs`)与 `ALTER TABLE low_confidence_questions` 加两列 + 索引 + 外键,逐字进 `db/schema.sql`(含 `SET NAMES utf8mb4` 守卫、ENGINE=InnoDB CHARSET=utf8mb4)。列要点:

- `review_queue`:id / normalized_question VARCHAR(512) / ai_suggested_answer TEXT / occurrence_count INT UNSIGNED 默认1 / review_status ENUM(待审·通过·驳回)默认待审 / approved_answer TEXT / created_at / updated_at。
- `eval_runs`:id / triggered_by ENUM(定时·手动)默认定时 / dataset_size INT UNSIGNED / metrics JSON / created_at。
- `low_confidence_questions` 加:`retrieved_chunks` JSON、`matched_review_id` BIGINT FK→review_queue ON DELETE SET NULL、`idx_matched_review_id`。

### 5.2 Langfuse 集成 `app/observability.py`

- config 加 `langfuse_secret_key` / `langfuse_public_key` / `langfuse_host`(默认 `http://localhost:3000`,未配 secret 则整条链路静默跳过)。
- 本章交付 `langfuse/docker-compose.yml`(自托管 Langfuse),你本地 `docker compose up` 起;SDK `langfuse` 加进 `requirements.txt`。
- `get_langfuse_handler() -> CallbackHandler | None`:env 未配返回 None;否则 `from langfuse.langchain import CallbackHandler`(v3 路径)惰性建。
- chat_service `_stream_invoke`:invoke/resume 的 `config` 加 `callbacks=[handler]` + `metadata={"langfuse_user_id": user_id, "langfuse_session_id": str(conv_id)}`;图跑完后用 handler 的 trace id 把 `intent:{最终意图}` 写回 trace tags。
- 异步风险(WebSearch 已确认):v3 `CallbackHandler` 仍是同步实现,async 图里可能阻塞/漏记。实现时优先验证;若 astream 下 trace 不完整,退而用 `@observe` 包住 graph 入口(记一条顶层 trace)。

### 5.3 evidence_confidence 闸 `app/graph/confidence.py`

- `compute_evidence_confidence(chunks) -> EvidenceConfidence(confidence: float, signals: dict)`。
- 信号:精排 Top1 相关性分、Top1−Top2 分差、有效证据条数(score ≥ 阈值 的 chunk 数)。
- 权重/阈值用 ch04 评估集校准(§5.9 的标定脚本),闸位置不动:`gate_node` 改用 evidence_confidence;拦下→`fallback_node` 落池(source=retrieval_low_conf + 快照 + signals 写 reason)。

### 5.4 三入口落池

统一走 `log_low_confidence(...)`,扩展参数:`retrieved_chunks`(快照)、`evidence_signals`(写进 reason 或单独列)。
1. 检索低置信(闸):`fallback_node` 已有,补快照 + signals。
2. 生成自评不足(新 `self_check` 节点):agent 出答案后,`self_check(llm, resolved_query, evidence)` 判 sufficient;不足→落池(source=self_check)。只落池,不改/不拦已流出的答案。
3. 👎 反馈:新 `POST /api/feedback`(conversation_id + 当轮用户问题文本)→ 落池(source=user_feedback),快照由**对该问题文本重跑一次检索**得到(尽力回捞,可能与原轮略不同);重跑也无结果则 None(闲聊/业务数据类本来就没检索)。

### 5.5 飞轮流水线 `app/jobs/flywheel.py`

处理 `low_confidence_questions` 里 `matched_review_id IS NULL` 的行:
1. **标准化**:LLM prompt 把口语原话改成标准 FAQ 式问题 + 一条示例答案(非可单测,评估集/标注样例验证)。
2. **查重**:LLM 判断与 `review_queue` 现有「待审/通过」问题是否同义;同义→`occurrence_count += 1` + 回写该行 `matched_review_id`;不同→新建 review_queue 行 + 回写。
- 手动触发:CLI `python -m app.jobs.flywheel` + 后台 job 端点复用 `/api/jobs/{name}`。

### 5.6 审核通过 → 落库

`POST /api/review/{id}/approve`(body 带 `approved_answer`):`review_status=通过` + `approved_answer`,并复用 ch03 的 chunk 落库(写 `knowledge_chunks` 一行 + 触发向量化,content_type 标 `faq`)。

### 5.7 评估流水线 `app/jobs/eval_run.py`

- `run_and_record(triggered_by) -> dict`:跑检索段(Recall@K、MRR)+ 生成段(Faithfulness),汇总成 metrics dict,写 `eval_runs` 一行,返回 metrics。
- CLI `python -m app.jobs.eval_run --triggered-by 手动`;趋势靠 `eval_runs.created_at` 连起来。

### 5.8 前端(Vibe Coding)

- `/review` 后台页 + `/api/review/*`:列表(标准化问题/出现次数/示例答案/状态)、详情(归并进来的原话 + 召回片段快照)、通过/驳回。

### 5.9 标定(evidence_confidence 阈值)

- 脚本 `scripts/calibrate_confidence.py`:在 ch04 评估集上跑检索,对每个 query 算三信号,扫权重/阈值组合,输出使「命中/未命中」分离度最好的一组,写进 config 或常量。

## 6. 留给 builder 的细节

- Langfuse v3 `CallbackHandler` 的确切导入路径、`config["metadata"]` 里 `langfuse_user_id/session_id/tags` 的确切键名、以及「事后回写 trace tags」的 SDK 调用(实现时 WebSearch/源码核实)。
- 异步回调在 `astream` 下的实际表现(是否需 `@observe` 兜底)。
- evidence_confidence 三信号的权重函数形式与标定结果。
- 查重/标准化的 LLM prompt 细节与标注样例。

## 7. 验收映射

1. Langfuse 点开任意请求看完整链路 → §5.2(env + 回调 + trace)。
2. 问知识库没有的问题→兜底→待审队列出现→详情看到原话+召回片段 → §5.3/5.4/5.6。
3. 后台点通过→再问同问题能答对 → §5.6。
4. 点 👎→落池→标准化查重→待审队列 → §5.4/5.5。
5. 按意图 token 花销统计 → §5.2(意图 tag 聚合)。
6. 评估流水线跑两轮看趋势 → §5.7。
