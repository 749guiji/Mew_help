"""Tests for ch10 synthetic-generation JSON parsing robustness."""

from ml.data.generate_synthetic import _normalize_label, _parse


def test_parse_clean_array():
    text = '[{"text": "买大了想退", "labels": ["尺码", "退换货"]}, {"text": "快递到哪了", "labels": ["物流"]}]'
    items = _parse(text)
    assert len(items) == 2
    assert items[0]["labels"] == ["尺码", "退换货"]


def test_parse_fenced_json():
    text = '```json\n[{"text": "要开发票", "labels": ["发票"]}]\n```'
    items = _parse(text)
    assert len(items) == 1


def test_parse_salvages_partial_items():
    # 数组没闭合/末尾截断,但前面几条是完整 JSON
    text = '[{"text": "运费怎么算", "labels": ["运费"]}, {"text": "能便宜点吗", "labels": ['
    items = _parse(text)
    assert len(items) == 1
    assert items[0]["text"] == "运费怎么算"


def test_normalize_label_fuzzy():
    assert _normalize_label("退货") == "退换货"
    assert _normalize_label(" 物流 ") == "物流"
    assert _normalize_label("乱七八糟") is None
