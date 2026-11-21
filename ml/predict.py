"""共享推理:模型/分词器 + 文本 → 多标签(供评测 / 服务 / 批量 job 复用)。"""

from __future__ import annotations

import torch

from ml.taxonomy import OTHER, topic_names


def predict_probs(model, tokenizer, texts: list[str], max_len: int = 128, device="cpu"):
    model.eval()
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


def labels_from_probs(probs, threshold: float = 0.5) -> list[list[str]]:
    names = topic_names()
    out: list[list[str]] = []
    for p in probs:
        ls = [names[i] for i in range(len(names)) if p[i] >= threshold]
        if not ls:
            ls = [OTHER]  # 全低于阈值 → 归「其他」
        out.append(ls)
    return out


def predict(model, tokenizer, texts: list[str], threshold: float = 0.5) -> list[list[str]]:
    probs = predict_probs(model, tokenizer, texts)
    return labels_from_probs(probs, threshold)
