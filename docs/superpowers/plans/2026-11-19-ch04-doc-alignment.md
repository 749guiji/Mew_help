# 对齐「混合检索与 RRF 融合」文档三处缺口 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已实现的 ch04 与用户提供的理论文档完全对齐，补齐负面清单、Lost-in-the-Middle 首尾重排、评估体系（证据覆盖度 / 答案覆盖度 / 拒答率）三处缺口。

**Architecture:** 前 5 个任务都是纯函数/数据/小改动，各自带单测；第 6 个任务把三块评估接入 `evaluation/run_eval.py` 脚本并用运行命令验证。沿用现有分层：纯指标在 `evaluation/metrics.py`，结构化 LLM 判定在 `evaluation/generation.py`（镜像 `faithfulness.py`），证据/引用组装在 `app/services/generation_quality.py`。

**Tech Stack:** Python 3.14、pydantic、langchain_core、pytest；本地 Ollama `qwen3:8b`。

**Spec:** `docs/superpowers/specs/2026-11-19-ch04-doc-alignment-design.md`

## Global Constraints

- 不引入文档「课程不做」的东西：HyDE、NLI 后处理、动态权重分数级融合。
- 保留 `ground_truth_chunk_ids`，不改成文档的 `expect_section`。
- LLM 类指标（答案覆盖度 / 拒答率）做成可选 flag，不默认跑。
- 引用编号 `[n]` 与 `citations` 的 `n` 必须指向同一份（重排后的）证据顺序。
- 结构化 LLM 调用一律 `with_structured_output(..., method="function_calling")`（langchain 1.x 默认 `json_schema` 在 Ollama 上静默失败）。

## Review Focus

1. **重排对 0/1/2 条证据不重排、不崩溃** → Task 2 测试。
2. **重排后 `format_evidence` 的 `[n]` 与 `build_citations` 的 `n` 指向同一 chunk** → Task 2 测试。
3. **`evidence_coverage` 空 `expect_points` / 空命中不除零** → Task 3 测试。
4. **`should_refuse=true` 的项 ground_truth 与 expect_points 必须为空、且只能落在 `D_refuse` 桶**（否则检索指标会被污染/除零）→ Task 4 校验测试。
5. **答案覆盖度判定在 FC-miss（返回 None）时不崩**，harness 跳过该条而非编造分数 → Task 5 测试。

---

### Task 1: 负面清单补齐

**Files:**
- Modify: `app/prompts/system.py`（`SYSTEM_PROMPT` 最后一条）
- Test: `tests/test_prompts.py`（新建）

**Interfaces:**
- Produces: `SYSTEM_PROMPT` 含文档红线三条 + 已有「退款到账时间」。

- [ ] **Step 1: 写失败测试**

`tests/test_prompts.py` 断言 `SYSTEM_PROMPT` 包含 `"具体到货时间"`、`"第三方物流时效"`、`"退款金额的最终裁定"`、`"退款到账时间"` 四个子串。

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL（后三个子串当前不在 `SYSTEM_PROMPT`）。

- [ ] **Step 3: 改 `SYSTEM_PROMPT` 最后一条**

把 `- 不承诺退款到账的具体时间。` 改为：
`- 红线，一律不承诺：具体到货时间、第三方物流时效、退款金额的最终裁定、退款到账时间。`

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/prompts/system.py tests/test_prompts.py
git commit -m "fix(ch04): 补齐负面清单红线(到货时间/第三方物流时效/退款金额裁定)"
```

---

### Task 2: Lost-in-the-Middle 首尾重排

**Files:**
- Modify: `app/services/generation_quality.py`（新增 `reorder_for_generation`，`format_evidence` 与 `build_citations` 入口调用）
- Test: `tests/test_generation_quality.py`

**Interfaces:**
- Consumes: `app.kb.hybrid.ScoredChunk`
- Produces: `reorder_for_generation(chunks: list[ScoredChunk]) -> list[ScoredChunk]`

- [ ] **Step 1: 写失败测试**

在 `tests/test_generation_quality.py` 增加：
- `test_reorder_places_top_two_at_ends`：5 个 chunk（id 1..5，分降序）→ `reorder_for_generation` 返回 id 序 `[1, 3, 5, 4, 2]`。
- `test_reorder_leaves_two_or_fewer_unchanged`：`[]`、单元素、双元素原序不变。
- `test_format_evidence_and_citations_share_numbering`：3 个 chunk（id 1/2/3），`format_evidence(chunks)` 第一行以 `[1]` 开头，`build_citations(chunks)` 的 `n` 序列与其 `chunk_id` 顺序一致（重排后应为 `[1, 3, 2]`）。

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_generation_quality.py -v`
Expected: FAIL（`reorder_for_generation` 未定义；重排断言不成立）。

- [ ] **Step 3: 实现 `reorder_for_generation` 并接入两函数**

`reorder_for_generation(chunks)`：`len <= 2` 返回 `list(chunks)`；否则返回 `chunks[0::2] + chunks[1::2][::-1]`。
在 `format_evidence` 与 `build_citations` 函数体第一行加 `chunks = reorder_for_generation(chunks)`。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_generation_quality.py -v`
Expected: PASS（含未改动的原有用例，因它们 ≤2 条不触发重排）

- [ ] **Step 5: Commit**

```bash
git add app/services/generation_quality.py tests/test_generation_quality.py
git commit -m "feat(ch04): Lost-in-the-Middle 首尾重排 + 引用编号一致性"
```

---

### Task 3: 证据覆盖度纯函数

**Files:**
- Modify: `evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `evidence_coverage(hit_texts: list[str], expect_points: list[str]) -> float`

- [ ] **Step 1: 写失败测试**

`tests/test_metrics.py` 增加：
- 全部命中 → `1.0`
- 部分命中 → `0.5`（两点命中其一）
- `expect_points == []` → `0.0`
- `hit_texts == []` → `0.0`

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL（`evidence_coverage` 未定义）

- [ ] **Step 3: 实现 `evidence_coverage(hit_texts, expect_points) -> float`**

`expect_points` 为空返回 `0.0`；否则把 `hit_texts` 拼成一个串，统计每个非空 `p` 是否 `p in blob`，返回命中数 / 总数。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(ch04): 证据覆盖度纯函数 evidence_coverage"
```

---

### Task 4: 评估集扩展（expect_points / should_refuse / D_refuse 桶）

**Files:**
- Modify: `evaluation/eval_set.jsonl`
- Modify: `evaluation/run_eval.py`（`BUCKETS` 加 `"D_refuse"`）
- Test: `tests/test_eval_set.py`（新建）

**Interfaces:**
- Consumes: `evaluation.run_eval.load_eval_set`、`evaluation.run_eval.BUCKETS`
- Produces: 每条 item 含 `expect_points: list[str]`、`should_refuse: bool`；`D_refuse` 桶约 10 条。

- [ ] **Step 1: 写失败测试**

`tests/test_eval_set.py` 从 `evaluation/eval_set.jsonl` 载入：
- 每条都含 `expect_points`（list）与 `should_refuse`（bool）。
- `should_refuse=true` 的项：`expect_points == []`、`ground_truth_chunk_ids == []`、`bucket == "D_refuse"`。
- `should_refuse=false` 的项：`expect_points` 非空、`bucket != "D_refuse"`。
- 所有 `bucket ∈ BUCKETS`，且 `"D_refuse" in BUCKETS`。

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_eval_set.py -v`
Expected: FAIL（字段缺失、无 D_refuse 桶）

- [ ] **Step 3: 扩评估集 + 加 D_refuse 桶**

给现有 50 条各加 `expect_points`（1–3 个**金标 chunk 答案里的逐字子串**）与 `should_refuse: false`。`expect_points` 按下表（`section_path` → 要点）取，映射时以 chunk 的 `section_path` 为准（chunk 17–36 已由现有 `ground_truth_chunk_ids` 定位，多跳 `E_multi` 取两段要点的并集）：

| section_path（节选） | expect_points |
|---|---|
| 七天无理由退货 | 签收之日起 7 天内 / 保持完好 / 不影响二次销售 |
| 运费说明 | 满 99 元包邮 / 8 元基础运费 / 偏远地区需额外加收 |
| 质量问题退货 | 运费由商家承担 / 拍照留证 |
| 退货流程 | 申请售后 / 审核通过 / 寄回商品 |
| 退款到账时间 | 1-3 个工作日 / 原路退回 / 以支付渠道为准 |
| 换货规则 | 支持换货 / 保持商品完好 |
| 发票开具 | 确认收货后自动开具 / 纸质发票需下单时勾选 |
| SKU-1001 | 29.9 / 硅胶防摔手机壳 / 现货 |
| SKU-1002 | 79 / 15W / Qi 协议 |
| SKU-1003 | 199 / 预售 / 蓝牙 5.3 |
| SKU-1004 | 89 / 10000mAh / 22.5W |
| 商品 FAQ·发货 | 48 小时 / 现货 |
| 商品 FAQ·发票 | 下单时勾选 / 补开发票 |
| 商品 FAQ·质量问题 | 拍照留证 / 联系在线客服 |

> 实现时先跑 `python -c "from app.db import SessionLocal; from app.db.models import KnowledgeChunk; s=SessionLocal(); [print(r.id, r.section_path, '|', r.answer[:60]) for r in s.query(KnowledgeChunk).order_by(KnowledgeChunk.id)]"` 确认 chunk→section 对应后，再把每条的 `expect_points` 写成该 chunk `answer` 里的逐字子串（保证字面匹配可复现）。

新增 `D_refuse` 桶 10 条（`ground_truth_chunk_ids: []`、`expect_points: []`、`should_refuse: true`、`difficulty` 给 `refuse`），query 如下：
`这个商品保修期多久？`、`支持以旧换新吗？`、`能用花呗分期付款吗？`、`有没有会员折扣或优惠券？`、`支持货到付款吗？`、`跨境订单退款多久到账？`、`可以自定义刻字吗？`、`有实体门店可以自提吗？`、`买两个有没有赠品？`、`海外可以发货吗？`

`BUCKETS` 改为 `("A_policy", "B_model", "C_colloquial", "D_refuse", "E_multi")`。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_eval_set.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evaluation/eval_set.jsonl evaluation/run_eval.py tests/test_eval_set.py
git commit -m "feat(ch04): 评估集补 expect_points/should_refuse + D_refuse 拒答桶"
```

---

### Task 5: 答案覆盖度 LLM-as-Judge

**Files:**
- Modify: `app/prompts/system.py`（新增 `ANSWER_COVERAGE_SYSTEM_PROMPT`）
- Create: `evaluation/generation.py`
- Test: `tests/test_generation.py`（新建，复用 `test_generation_quality.py` 的 `FakeLLM` 模式）

**Interfaces:**
- Produces: `AnswerCoverageResult(BaseModel)`（`covered: list[int]`，0 起下标）；`judge_answer_coverage(llm, question, answer, expect_points) -> AnswerCoverageResult | None`

- [ ] **Step 1: 写失败测试**

`tests/test_generation.py`：
- `test_judge_answer_coverage_returns_covered`：`FakeLLM(AnswerCoverageResult(covered=[0, 2]))` → 返回 `covered == [0, 2]`，且 `llm.called_method == "function_calling"`。
- `test_judge_answer_coverage_returns_none_on_fc_miss`：`FakeLLM(None)` → 返回 `None`。

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_generation.py -v`
Expected: FAIL（模块/函数未定义）

- [ ] **Step 3: 实现判定函数 + prompt**

`ANSWER_COVERAGE_SYSTEM_PROMPT`：给定用户问题、助手回答、编号的标准答案要点（从 0 起编号），返回 `covered`（被回答正确覆盖的要点下标列表，用 `method="function_calling"` 结构化输出）。
`judge_answer_coverage`：拼 prompt → `with_structured_output(AnswerCoverageResult, method="function_calling")` → `isinstance` 检查，命中返回实例，否则返回 `None`（与 `self_check` 回退方向相反：这是度量不是闸，FC-miss 交由 harness 跳过，不编造分数）。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/prompts/system.py evaluation/generation.py tests/test_generation.py
git commit -m "feat(ch04): 答案覆盖度 LLM-as-Judge judge_answer_coverage"
```

---

### Task 6: 评估脚本接线（证据覆盖度 + 答案覆盖度 + 拒答率）

**Files:**
- Modify: `evaluation/run_eval.py`

**Interfaces:**
- Consumes: `evidence_coverage`（Task 3）、`judge_answer_coverage`/`AnswerCoverageResult`（Task 5）、`default_retrieve`、`format_evidence`、`self_check`、`_generate_answer`、`run_strategy`（现有）
- Produces: `run_eval.py` 输出含证据覆盖度的四策略表；`--answer-coverage`、`--refusal` 两个 flag。

- [ ] **Step 1: 检索报告接入证据覆盖度**

在 `main()` 的逐项统计循环里，对非 `D_refuse` 的 item，把 `[c.answer for c in chunks]` 作为 `hit_texts`、`item["expect_points"]` 传入 `evidence_coverage`，按 `(strategy, bucket)` 累计；打印表头加 `EVCOV` 列（与 R@k/MRR 并列）。`D_refuse` 桶不参与检索指标（无 ground truth）。

- [ ] **Step 2: `--answer-coverage` flag**

新增布尔 flag `--answer-coverage`。开启时，对每个非 `D_refuse` item，按 `--strategies`（默认四策略，`hybrid_rerank` 缺 torch 时跳过）用 `run_strategy` 取 chunks → `_generate_answer` 生成答案 → `judge_answer_coverage` 判定，覆盖度 = `len(set(i for i in covered if 0 <= i < n)) / n`（`n = len(expect_points)`，`n==0` 跳过；判定返回 `None` 计入 errors 跳过）。按 strategy 汇总打印「答案覆盖度」，支持 `--limit`。

- [ ] **Step 3: `--refusal` flag**

新增布尔 flag `--refusal`。开启时，对 `D_refuse` 桶每条：`default_retrieve(query)` → 无结果则「拒答」；否则 `self_check(llm, query, format_evidence(chunks))` → `sufficient=false` 则「拒答」，`true` 则「未拒答」。打印「拒答率 = 拒答数 / 总数」，支持 `--limit`。

- [ ] **Step 4: 验证检索报告**

Run: `pytest -q`
Expected: 全绿（原 140 例 + 新增）。

Run: `python evaluation/run_eval.py --strategies dense,sparse,hybrid,hybrid_rerank`
Expected: 每策略 × 桶打印 R@1/3/5/10、MRR、**EVCOV**；`D_refuse` 不出现检索指标行；`B_model` 桶 EVCOV 接近 1.0（型号是逐字点）。

- [ ] **Step 5: 验证两个 flag**

Run: `python evaluation/run_eval.py --answer-coverage --limit 10`
Expected: 打印答案覆盖度（按策略汇总），无崩溃。

Run: `python evaluation/run_eval.py --refusal`
Expected: 打印拒答率，`D_refuse` 10 条应接近 1.0。

- [ ] **Step 6: Commit**

```bash
git add evaluation/run_eval.py
git commit -m "feat(ch04): 检索报告加证据覆盖度 + 答案覆盖度/拒答率 flag"
```
