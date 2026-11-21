"""文本清洗:脱敏(手机号/订单号)+ 修格式。"""

from __future__ import annotations

import re

_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_ORDER_RE = re.compile(r"(?:订单|单号|尾号)\s*\d{4,}")


def clean(text: str) -> str:
    """脱敏手机号、订单号,折叠多余空白,去首尾。"""
    text = _PHONE_RE.sub("<手机号>", text)
    text = _ORDER_RE.sub("<订单号>", text)
    text = " ".join(text.split())
    return text.strip()
