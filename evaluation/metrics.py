"""Pure retrieval metrics for the ch04 eval harness.

Both functions take a ranked list of retrieved chunk ids (``hit_ids``, best
first) and the ground-truth chunk ids for one eval item. They are deliberately
free of DB / vector-store / model dependencies so they can be unit tested and
reused by both the retrieval and generation eval sections.
"""

from __future__ import annotations


def recall_at_k(hit_ids: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of ground-truth ids found within the top-``k`` hits.

    ``k <= 0`` or an empty ground truth yields ``0.0`` (no relevant docs to
    find, so the denominator would otherwise be zero). ``k`` larger than the
    hit list simply uses the whole list.
    """
    if k <= 0 or not ground_truth:
        return 0.0
    retrieved = set(hit_ids[:k])
    return len(retrieved.intersection(ground_truth)) / len(ground_truth)


def mrr(hit_ids: list[int], ground_truth: list[int]) -> float:
    """Mean-reciprocal-rank for a single query: ``1 / rank`` of the first hit.

    Returns ``0.0`` when no ground-truth id appears in the ranked list.
    """
    gt = set(ground_truth)
    for rank, chunk_id in enumerate(hit_ids, start=1):
        if chunk_id in gt:
            return 1.0 / rank
    return 0.0
