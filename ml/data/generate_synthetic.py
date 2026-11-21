"""大模型生成 17 类多标签模拟样本(飞轮池空,补足 1700)。

每类批量生成,小批量(默认 10)更稳;JSON 解析宽松(去围栏/逐项兜底)+ 标签归一
(去空白/精确/子串);某类不够再补跑几批。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_llm
from ml.taxonomy import TOPICS, topic_names

OUT = Path("ml/data/raw_synthetic.jsonl")
PER_CLASS = 100
BATCH = 10
MAX_EXTRA = 5  # 每类最多补跑批数

_NAMES = topic_names()
_NAME_SET = set(_NAMES)

# 大模型常把类名写偏,这里显式归一下
_ALIASES = {
    "退货": "退换货", "退款": "退换货", "换货": "退换货", "退货退款": "退换货",
    "快递": "物流", "配送": "物流",
    "尺寸": "尺码", "大小": "尺码",
    "优惠券": "优惠活动", "满减": "优惠活动", "折扣": "优惠活动",
    "维修": "保修维修", "保修": "保修维修",
    "积分": "会员积分", "会员": "会员积分",
}


def _prompt(name: str, boundary: str, n: int) -> str:
    lines = "\n".join(f"- {t['name']}：{t['boundary']}" for t in TOPICS)
    return (
        f"你是电商售后客服语料生成器。生成 {n} 条真实用户提问,"
        f"主主题覆盖「{name}」({boundary}),并自然穿插其他主题。\n"
        f"多标签规则:一句字面提到几个诉求就标几个标签,一个不多一个不少;"
        f"标签必须一字不差地用下面 17 个类名。\n\n17 类:\n{lines}\n\n"
        f"只输出 JSON 数组,每项 {{\"text\":\"<问题>\",\"labels\":[\"<类名>\",...]}}。"
        f"不要代码围栏、不要解释。"
    )


def _normalize_label(l) -> str | None:
    if not isinstance(l, str):
        return None
    l = l.strip()
    if l in _NAME_SET:
        return l
    if l in _ALIASES:
        return _ALIASES[l]
    for name in _NAMES:  # 子串兜底
        if l in name or name in l:
            return name
    return None


def _valid(data) -> list[dict]:
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        t, labels = it.get("text"), it.get("labels")
        if not isinstance(t, str) or not isinstance(labels, list):
            continue
        labels = [nl for l in labels if (nl := _normalize_label(l))]
        if t.strip() and labels:
            out.append({"text": t.strip(), "labels": labels})
    return out


def _parse(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) > 2:
            text = "\n".join(lines[1:-1])
    # 先整体解析(完整数组)
    try:
        return _valid(json.loads(text))
    except Exception:
        pass
    # 兜底:逐项抠 { ... }(应对尾部截断)
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            it = json.loads(m.group(0))
        except Exception:
            continue
        t, labels = it.get("text"), it.get("labels")
        if isinstance(t, str) and isinstance(labels, list):
            labels = [nl for l in labels if (nl := _normalize_label(l))]
            if t.strip() and labels:
                out.append({"text": t.strip(), "labels": labels})
    return out


def generate(per_class: int = PER_CLASS, batch: int = BATCH) -> int:
    llm = get_llm(get_settings())
    written = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for t in TOPICS:
            got = 0
            attempts = per_class // batch + MAX_EXTRA
            for _ in range(attempts):
                if got >= per_class:
                    break
                try:
                    resp = llm.invoke(_prompt(t["name"], t["boundary"], batch))
                    items = _parse(resp.content)
                except Exception as e:  # noqa: BLE001
                    print(f"[error] {t['name']}: {e}", flush=True)
                    continue
                for it in items:
                    fh.write(json.dumps(it, ensure_ascii=False) + "\n")
                    got += 1
                print(f"{t['name']}: +{len(items)} (累计 {got}/{per_class})", flush=True)
            written += got
    print(f"done: {written} 条 -> {OUT}")
    return written


if __name__ == "__main__":
    generate()
