"""Tests for prompt constants (negative list / red lines)."""

from app.prompts.system import SYSTEM_PROMPT


def test_negative_list_covers_doc_red_lines():
    for phrase in (
        "具体到货时间",
        "第三方物流时效",
        "退款金额的最终裁定",
        "退款到账时间",
    ):
        assert phrase in SYSTEM_PROMPT, f"missing red line: {phrase}"
