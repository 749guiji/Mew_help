"""Tests for ch10 data pipeline (clean / stratified split / augment)."""

from ml.data.augment import augment
from ml.data.clean import clean
from ml.data.split import stratified_split


def test_clean_masks_phone_and_order():
    assert "13812345678" not in clean("我手机13812345678想退")
    assert "1001" not in clean("订单1001的物流到哪了")
    assert "尾号20260701" not in clean("尾号20260701那单")


def test_clean_collapses_whitespace():
    assert clean("  运费   怎么算  ") == "运费 怎么算"


def test_split_preserves_ratio_and_total():
    items = [
        {"text": f"q{i}", "labels": ["物流"] if i % 2 == 0 else ["退换货"]}
        for i in range(100)
    ]
    train, val, test = stratified_split(items, seed=42)
    assert len(train) + len(val) + len(test) == 100
    assert len(train) == 80 and len(val) == 10 and len(test) == 10
    # 每份两个类都还在(保比例)
    for part in (train, val, test):
        labels = {l for it in part for l in it["labels"]}
        assert labels == {"物流", "退换货"}


def test_augment_produces_variant():
    out = augment("运费怎么算")
    assert len(out) >= 1
    assert "运费怎么算" not in out  # 不含原文
