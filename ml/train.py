"""RoBERTa-wwm-ext 全参微调(多标签 BCE + 早停 + checkpoint 断点续训)。CPU 上跑。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer

from ml.model import TopicClassifier
from ml.predict import predict
from ml.taxonomy import num_labels, topic_names

MODEL_PATH = "ml/artifacts/model/models/dienstag--chinese-roberta-wwm-ext/snapshots/v1.0.0"
DATA_DIR = Path("ml/data")
ARTIFACTS = Path("ml/artifacts")


class _DS(Dataset):
    def __init__(self, items, tokenizer, label2id, max_len=128):
        self.items = items
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        enc = self.tokenizer(
            it["text"], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt"
        )
        y = torch.zeros(num_labels)
        for l in it["labels"]:
            y[self.label2id[l]] = 1.0
        return enc["input_ids"][0], enc["attention_mask"][0], y


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def macro_f1(model, tokenizer, items, label2id, device="cpu"):
    if not items:
        return 0.0
    texts = [it["text"] for it in items]
    gold = [[1 if name in it["labels"] else 0 for name in topic_names()] for it in items]
    pred_labels = predict(model, tokenizer, texts)
    pred = [[1 if name in ls else 0 for name in topic_names()] for ls in pred_labels]
    return f1_score(gold, pred, average="macro", zero_division=0)


def train(epochs: int = 5, lr: float = 2e-5, wd: float = 0.01, batch_size: int = 16) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    label2id = {name: i for i, name in enumerate(topic_names())}

    train_items = _load(DATA_DIR / "train.jsonl")
    val_items = _load(DATA_DIR / "val.jsonl")
    if not train_items:
        raise SystemExit("无训练数据,先跑 ml/data/generate_synthetic.py 再 build_dataset")

    model = TopicClassifier(MODEL_PATH, num_labels)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    dl = DataLoader(_DS(train_items, tokenizer, label2id), batch_size=batch_size, shuffle=True)
    best_f1, best_epoch, patience = 0.0, -1, 0
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        t0 = time.time()
        for ids, mask, y in dl:
            opt.zero_grad()
            logits = model(ids, attention_mask=mask)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        val_f1 = macro_f1(model, tokenizer, val_items, label2id)
        avg_loss = total_loss / max(n, 1)
        print(f"epoch {ep+1}/{epochs} loss={avg_loss:.4f} val_macro_f1={val_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)

        torch.save({"model": model.state_dict(), "epoch": ep, "f1": val_f1}, ARTIFACTS / "last.pt")
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, ep, 0
            torch.save(model.state_dict(), ARTIFACTS / "best.pt")
        else:
            patience += 1
            if patience >= 2:
                print(f"早停 @ epoch {ep+1}", flush=True)
                break

    print(f"best val_macro_f1={best_f1:.4f} @ epoch {best_epoch+1}")
    return str(ARTIFACTS / "best.pt")


if __name__ == "__main__":
    train()
