"""evidence_confidence 阈值标定(ch09)。

在 ch04 评估集上,对每个 query 跑生产检索(default_retrieve),算三信号
(top1_score / top1_top2_gap / effective_count),再扫一组阈值,看哪个阈值让
「该答的(有 ground_truth)」过闸、「该拒的(should_refuse / 无 gt)」拦下的正确率最高。

用法(需先灌知识库):
    PYTHONPATH=. .venv/Scripts/python.exe scripts/calibrate_confidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.graph.confidence import compute_evidence_confidence  # noqa: E402
from app.jobs.eval_run import load_eval_set  # noqa: E402
from app.kb.retrieval import default_retrieve  # noqa: E402


def main() -> int:
    items = load_eval_set()
    rows = []
    for item in items:
        chunks = default_retrieve(item["query"]) or []
        r = compute_evidence_confidence(chunks)
        should_answer = bool(item.get("ground_truth_chunk_ids")) and not item.get(
            "should_refuse", False
        )
        rows.append((should_answer, r["confidence"]))

    if not rows:
        print("评估集为空;先灌知识库再跑。")
        return 1

    best = (None, -1.0)
    for t in [round(i * 0.05, 2) for i in range(21)]:  # 0.00 .. 1.00
        correct = sum(
            1 for should_answer, conf in rows if (conf >= t) == should_answer
        )
        acc = correct / len(rows)
        if acc > best[1]:
            best = (t, acc)
        print(f"threshold={t:.2f} accuracy={acc:.3f}")

    print(f"\n最优阈值 {best[0]}(准确率 {best[1]:.3f}),写进 app/graph/confidence.py 的 CONFIDENCE_THRESHOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
