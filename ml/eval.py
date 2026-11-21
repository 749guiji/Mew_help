"""评测:每类 precision/recall/F1 + 混淆矩阵,写 eval_report.json + errors.jsonl。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer

from ml.model import TopicClassifier
from ml.predict import predict
from ml.taxonomy import num_labels, topic_names

MODEL_PATH = "ml/artifacts/model/models/dienstag--chinese-roberta-wwm-ext/snapshots/v1.0.0"
ARTIFACTS = Path("ml/artifacts")


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(model_path: Path, data_path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = TopicClassifier(MODEL_PATH, num_labels)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    items = _load(data_path)
    names = topic_names()
    texts = [it["text"] for it in items]
    gold = [[1 if name in it["labels"] else 0 for name in names] for it in items]
    pred_labels = predict(model, tokenizer, texts)
    pred = [[1 if name in ls else 0 for name in names] for ls in pred_labels]

    p, r, f1, _ = precision_recall_fscore_support(gold, pred, average=None, zero_division=0)
    m = precision_recall_fscore_support(gold, pred, average="macro", zero_division=0)
    cm = multilabel_confusion_matrix(gold, pred)

    report = {
        "macro": {"precision": round(m[0], 4), "recall": round(m[1], 4), "f1": round(m[2], 4)},
        "per_class": [
            {"name": names[i], "precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f1[i], 4)}
            for i in range(num_labels)
        ],
        "confusion": [
            {"name": names[i], "tn": int(cm[i][0][0]), "fp": int(cm[i][0][1]),
             "fn": int(cm[i][1][0]), "tp": int(cm[i][1][1])}
            for i in range(num_labels)
        ],
    }

    errors = [
        {"text": it["text"], "gold": it["labels"], "pred": pl}
        for it, g, pl in zip(items, gold, pred_labels)
        if g != [1 if name in pl else 0 for name in names]
    ]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "errors.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in errors), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    r = evaluate(Path("ml/artifacts/best.pt"), Path("ml/data/test.jsonl"))
    print(json.dumps(r["macro"], ensure_ascii=False))
    for c in r["per_class"]:
        print(f"  {c['name']}: P={c['precision']} R={c['recall']} F1={c['f1']}")
