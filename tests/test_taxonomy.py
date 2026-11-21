"""Tests for ch10 17-class topic taxonomy (single source of truth)."""

from ml.taxonomy import OTHER, TOPICS, num_labels, topic_names


def test_17_topics_ordered():
    names = topic_names()
    assert len(names) == 17
    # 四大领头类在前,顺序固定
    assert names[:4] == ["退换货", "物流", "尺码", "发票"]
    assert OTHER in names


def test_num_labels_matches():
    assert num_labels == len(TOPICS) == 17


def test_hard_boundary_rules_preserved():
    b = {t["name"]: t["boundary"] for t in TOPICS}
    # 用户三条硬规则逐字保留
    assert "退" in b["退换货"]
    assert "修" in b["保修维修"]
    assert "钱" in b["运费"] and "货" in b["物流"]
    assert "差价" in b["价保"]
    assert ("满减" in b["优惠活动"]) or ("券" in b["优惠活动"])
