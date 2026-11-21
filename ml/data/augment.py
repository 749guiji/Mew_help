"""数据增强:同义词替换 + 句式微调(只扩训练集)。"""

from __future__ import annotations

_SYNONYMS = {
    "怎么": "咋",
    "吗": "么",
    "便宜": "实惠",
    "多少钱": "价格",
    "退款": "退货退款",
    "能不能": "能否",
}


def augment(text: str) -> list[str]:
    """返回若干变体(去重、排除原文)。"""
    variants: list[str] = []
    for a, b in _SYNONYMS.items():
        if a in text:
            variants.append(text.replace(a, b))
    variants.append(f"请问{text}")
    variants.append(text.rstrip("?？") + "呢")
    return [v for v in dict.fromkeys(variants) if v != text]
