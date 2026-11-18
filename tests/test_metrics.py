"""Tests for the pure retrieval metrics (ch04 Task 7).

``recall_at_k`` and ``mrr`` are pure functions over ranked ``hit_ids``; these
tests pin the exact arithmetic, including zero hits and out-of-range ``k``.
"""

import pytest

from evaluation.metrics import mrr, recall_at_k


def test_recall_full_hit():
    assert recall_at_k([1, 2, 3], [2], k=3) == 1.0


def test_recall_partial_overlap():
    # One of two ground-truth ids appears in the top-3 window.
    assert recall_at_k([1, 2, 3], [2, 9], k=3) == 0.5


def test_recall_k_narrows_window():
    # The relevant id sits at rank 2, so a k=1 window misses it.
    assert recall_at_k([1, 2, 3], [2, 9], k=1) == 0.0


def test_recall_k_out_of_range_clamps_to_all_hits():
    # k larger than the hit list degrades to the full list.
    assert recall_at_k([1, 2], [2], k=10) == 1.0


def test_recall_zero_hits():
    assert recall_at_k([], [1], k=5) == 0.0


def test_recall_empty_ground_truth():
    # No relevant docs to find: define recall as 0 rather than divide by zero.
    assert recall_at_k([1, 2], [], k=5) == 0.0


def test_recall_zero_or_negative_k():
    assert recall_at_k([1, 2, 3], [1], k=0) == 0.0
    assert recall_at_k([1, 2, 3], [1], k=-1) == 0.0


def test_mrr_first_position():
    assert mrr([1, 2, 3], [1]) == 1.0


def test_mrr_second_position():
    assert mrr([1, 2, 3], [2]) == 0.5


def test_mrr_third_position():
    assert mrr([1, 2, 3], [3]) == pytest.approx(1.0 / 3.0)


def test_mrr_no_hit():
    assert mrr([1, 2, 3], [9]) == 0.0


def test_mrr_empty_hits():
    assert mrr([], [1]) == 0.0


def test_mrr_multiple_ground_truth_takes_first_hit():
    # Only the earliest relevant rank counts for MRR.
    assert mrr([1, 2, 3], [2, 3]) == 0.5
