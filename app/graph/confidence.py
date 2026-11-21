"""Evidence-confidence gate (ch09).

把 ch05 的「retrieval_score >= threshold」最简闸升级成正式置信度:用精排
Top1 相关性分、Top1−Top2 分差、有效证据条数三个信号加权。权重/阈值在
``scripts/calibrate_confidence.py`` 用 ch04 评估集校准后回填到常量。
"""

from __future__ import annotations

from app.kb.hybrid import ScoredChunk

# 三信号权重(初值,Task 10 用评估集校准后回填)
_W_TOPK1 = 0.5
_W_EFFECTIVE = 0.3
_W_GAP = 0.2

# 有效证据的分数下限 + 归一化锚点
_EFFECTIVE_SCORE_FLOOR = 0.5
_EFFECTIVE_COUNT_ANCHOR = 3
_GAP_ANCHOR = 0.5

# 知识类闸的置信度阈值(初值,Task 10 校准)
CONFIDENCE_THRESHOLD = 0.5


def compute_evidence_confidence(chunks: list[ScoredChunk]) -> dict:
    """算知识类检索的置信度与信号。

    返回 ``{"confidence": float, "signals": {...}}``。空检索 → confidence 0.0。
    """
    if not chunks:
        return {
            "confidence": 0.0,
            "signals": {"top1_score": 0.0, "top1_top2_gap": 0.0, "effective_count": 0},
        }

    scores = [c.score for c in chunks]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else None
    gap = (top1 - top2) if top2 is not None else 0.0  # 无 Top2 → 无分差可言,不给满分差
    effective = sum(1 for s in scores if s >= _EFFECTIVE_SCORE_FLOOR)

    gap_norm = min(max(gap, 0.0) / _GAP_ANCHOR, 1.0)
    eff_norm = min(effective / _EFFECTIVE_COUNT_ANCHOR, 1.0)

    confidence = _W_TOPK1 * top1 + _W_EFFECTIVE * eff_norm + _W_GAP * gap_norm
    return {
        "confidence": round(confidence, 4),
        "signals": {
            "top1_score": round(top1, 4),
            "top1_top2_gap": round(gap, 4),
            "effective_count": effective,
        },
    }
