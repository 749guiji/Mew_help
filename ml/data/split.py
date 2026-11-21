"""分层抽样:按标签集分组,每组 8:1:1 进 train/val/test。"""

from __future__ import annotations

import random


def stratified_split(items: list[dict], seed: int = 42) -> tuple[list, list, list]:
    """多标签分层:同标签组合成组,每组按 80/10/10 划。"""
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for it in items:
        key = tuple(sorted(it["labels"]))
        groups.setdefault(key, []).append(it)

    train, val, test = [], [], []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])
    return train, val, test
