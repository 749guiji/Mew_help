# 设计：对齐「混合检索与 RRF 融合」文档三处缺口

日期：2026-11-19
范围：把已实现的 ch04 与用户提供的理论文档完全对齐，补齐三处未落实的缺口。

## 意图与原因

ch04（混合检索 + RRF + 重排 + 评估）已实现并合并 main，与文档高度吻合。逐条对照后发现三处文档明确要求、但代码尚未落实：

1. **负面清单不完整** —— 文档红线三条（具体到货时间 / 第三方物流时效 / 退款金额的最终裁定），当前 `SYSTEM_PROMPT` 只写了「不承诺退款到账的具体时间」。
2. **Lost in the Middle 未做** —— 文档要求精排后「最相关的摆最前**和最后**」，当前 `format_evidence` 只按分数降序（最好的在前、次好的不挪到尾）。
3. **评估体系不完整** —— 缺证据覆盖度、答案覆盖度、拒答率三个指标；评估集缺 `expect_points` / `should_refuse` 字段；缺「该拒答」桶。

## 目标 / 非目标 / 反目标

- 目标：三处缺口全部落地，测试全绿，四策略报告与生成段指标可复现跑出。
- 非目标：不引入文档明确「课程不做」的东西（HyDE、NLI 后处理、动态权重分数级融合）。
- 反目标：不为了对齐文档而重写已稳定工作的检索链路（`ground_truth_chunk_ids` 比文档的 `expect_section` 更精确，保留不动）。

## 约束

- 本地 Ollama `qwen3:8b`，LLM 类指标（答案覆盖度 / 拒答率）要做成可选 flag + `--limit`，避免默认跑太慢。
- 精排模型 `bge-reranker-v2-m3` 走 torch 进程内加载，`hybrid_rerank` 策略在缺 torch 时已有跳过逻辑，沿用。
- 引用编号 `[n]` 与 `citations` 的 `n` 必须指向同一份（重排后的）证据顺序，否则前端可点引用会错位。

## 方案

### 1. 负面清单补齐（`app/prompts/system.py`）

把 `SYSTEM_PROMPT` 最后一条改成文档的红线写法，并保留现有「退款到账时间」：

> - 红线，一律不承诺：具体到货时间、第三方物流时效、退款金额的最终裁定、退款到账时间。

### 2. Lost in the Middle 首尾重排（`app/services/generation_quality.py`）

新增纯函数 `reorder_for_generation(chunks) -> list[ScoredChunk]`：长度 ≤ 2 原样返回；否则「前偶后奇倒序」交错——`chunks[0::2] + chunks[1::2][::-1]`。对 `[1,2,3,4,5]` 得到 `[1,3,5,4,2]`（#1 最前、#2 最后、#5 居中）。

`format_evidence` 与 `build_citations` **都在入口先调用它**，保证两处编号一致（`query_faq` 工具、`run_eval` 忠实度段这两个调用点自动保持一致，无需改动）。

### 3. 评估体系补齐

**3a. 评估集扩展（`evaluation/eval_set.jsonl`）**
- 现有 50 条全部补 `expect_points`（标准答案拆出的要点，短串）与 `should_refuse: false`。
- 新增 `D_refuse` 桶约 10 条：知识库中没有、该拒答的问题，`should_refuse: true`、`ground_truth_chunk_ids=[]`、`expect_points=[]`。
- 桶集合改为 `("A_policy", "B_model", "C_colloquial", "D_refuse", "E_multi")`。

**3b. 证据覆盖度（`evaluation/metrics.py`，纯函数，无 LLM）**
`evidence_coverage(hit_texts: list[str], expect_points: list[str]) -> float`：每个 `expect_point` 若字面出现在任一 `hit_text` 中则算命中，返回命中率。可单测、可复现。接入检索报告（每个策略 × 桶，跳过 `D_refuse`）。

**3c. 答案覆盖度（LLM-as-Judge，`--answer-coverage` flag）**
新增 `ANSWER_COVERAGE_SYSTEM_PROMPT` + `AnswerCoverageResult {covered: list[int]}`（被覆盖的 `expect_points` 下标），对每个策略生成答案并判定覆盖度，`--strategies` 复用现有参数、默认四策略，跳过 `D_refuse`，支持 `--limit`。

**3d. 拒答率（`--refusal` flag）**
对 `D_refuse` 桶的题：`default_retrieve` → 无结果或 `self_check` 判 `sufficient=false` 计为正确拒答，算拒答率。复用生产路径的 `default_retrieve` + `self_check`，与线上逻辑同源。

## 我做的关键决断（自定，非用户拍板）

- 保留 `ground_truth_chunk_ids`，不改成文档的 `expect_section`（更精确、零迁移）。
- 重排逻辑用「前偶后奇倒序」，最简实现文档「首尾」意图。
- `D_refuse` 桶名 + 保留 `E_multi` 桶（多跳是现有有用桶，文档没禁止）。
- 答案覆盖度 / 拒答率做成独立可选 flag（非默认），契合本地模型延迟。
- 负面清单做成四条并集（文档三条 + 已有退款到账时间），不丢已有约束。

## 留给 builder 的

- 改动文件：`app/prompts/system.py`、`app/services/generation_quality.py`、`evaluation/metrics.py`、`evaluation/run_eval.py`、`evaluation/eval_set.jsonl`（新增 `evaluation/generation.py` 承载答案覆盖度/拒答率判定，若体积合适）。
- 测试：`test_generation_quality.py`（重排 + 编号一致性）、`test_metrics.py`（证据覆盖度边界）、`test_routes.py`/`test_static.py` 回归、新增 `SYSTEM_PROMPT` 红线断言。
- 验证：`pytest` 全绿；`python evaluation/run_eval.py --strategies dense,sparse,hybrid,hybrid_rerank` 出含证据覆盖度的四策略表；`--answer-coverage --limit N`、`--refusal` 各出数字。
