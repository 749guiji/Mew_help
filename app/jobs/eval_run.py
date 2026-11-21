"""评估流水线(ch09):复用 ch04 评估集跑检索段指标,落 eval_runs 看趋势。

每轮把 Recall@3 / MRR / 证据覆盖度汇总成 metrics 写进 eval_runs,按时间连成趋势。
生成段 Faithfulness 较重(逐条 LLM),暂不在默认轮次跑,留 retrieve_fn 注入点供扩展。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.db import SessionLocal
from app.db.models import EvalRun
from evaluation.metrics import evidence_coverage, mrr, recall_at_k

_EVAL_SET = Path("evaluation/eval_set.jsonl")


def load_eval_set(path=None) -> list[dict]:
    path = Path(path or _EVAL_SET)
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_retrieval_metrics(items, retrieve_fn) -> dict:
    """检索段指标:Recall@3 / MRR / 证据覆盖度(取全部 item 均值)。

    ``retrieve_fn(query) -> list[ScoredChunk]``。
    """
    sums = {"R@3": 0.0, "MRR": 0.0, "EVCOV": 0.0}
    n = 0
    for item in items:
        chunks = retrieve_fn(item["query"]) or []
        hit_ids = [c.chunk_id for c in chunks]
        gt = item.get("ground_truth_chunk_ids", [])
        sums["R@3"] += recall_at_k(hit_ids, gt, 3)
        sums["MRR"] += mrr(hit_ids, gt)
        sums["EVCOV"] += evidence_coverage(
            [c.answer for c in chunks], item.get("expect_points", [])
        )
        n += 1
    if not n:
        return {"recall_at_k": 0.0, "mrr": 0.0, "evidence_coverage": 0.0}
    return {
        "recall_at_k": round(sums["R@3"] / n, 4),
        "mrr": round(sums["MRR"] / n, 4),
        "evidence_coverage": round(sums["EVCOV"] / n, 4),
    }


def run_eval_and_record(
    triggered_by: str = "手动", limit: int | None = None, retrieve_fn=None
) -> dict:
    """跑一轮评估并把指标落 eval_runs;返回 metrics dict。"""
    if retrieve_fn is None:
        from app.kb.retrieval import default_retrieve

        retrieve_fn = default_retrieve
    items = load_eval_set()
    if limit:
        items = items[:limit]
    metrics = compute_retrieval_metrics(items, retrieve_fn)
    with SessionLocal() as s:
        er = EvalRun(triggered_by=triggered_by, dataset_size=len(items), metrics=metrics)
        s.add(er)
        s.commit()
    return metrics
