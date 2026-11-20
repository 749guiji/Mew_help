"""Robust JSON object extraction for LLM outputs (ch06).

qwen3:8b 的 function-calling 结构化输出不稳(ch05 已知坑),所以强制 JSON 走
「prompt 强制 + 稳健解析」:从模型输出里截取首个 ``{...}`` 块,容忍 markdown 代码
围栏与前后废话,``json.loads`` 失败则返回 ``None`` 让调用方兜底。
"""

from __future__ import annotations

import json


def parse_json_object(text: str) -> dict | None:
    """Extract and parse the first JSON object from ``text``, or ``None``.

    Handles the common LLM wrap (leading/trailing prose, markdown fences).
    Returns ``None`` when no object is present or it fails to parse.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start : end + 1].strip("`")
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None
