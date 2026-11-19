"""Schema validation for the ch04 eval set (expect_points / should_refuse)."""

from pathlib import Path

from evaluation.run_eval import BUCKETS, load_eval_set

_EVAL_SET = Path(__file__).resolve().parent.parent / "evaluation" / "eval_set.jsonl"


def _items():
    return load_eval_set(_EVAL_SET)


def test_every_item_has_points_and_refuse_flag():
    for it in _items():
        assert "expect_points" in it and isinstance(it["expect_points"], list)
        assert "should_refuse" in it and isinstance(it["should_refuse"], bool)


def test_refuse_items_are_empty_and_in_refuse_bucket():
    for it in _items():
        if it["should_refuse"]:
            assert it["expect_points"] == []
            assert it["ground_truth_chunk_ids"] == []
            assert it["bucket"] == "D_refuse"


def test_non_refuse_items_have_points():
    for it in _items():
        if not it["should_refuse"]:
            assert it["expect_points"], f"empty expect_points on {it['id']}"
            assert it["bucket"] != "D_refuse"


def test_all_buckets_are_known():
    assert "D_refuse" in BUCKETS
    for it in _items():
        assert it["bucket"] in BUCKETS, f"unknown bucket {it['bucket']}"
