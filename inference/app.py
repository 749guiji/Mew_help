"""独立主题分类推理服务(ONNX + onnxruntime,端口 8001)。"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

from ml.taxonomy import OTHER, topic_names

MODEL_PATH = "ml/artifacts/model/models/dienstag--chinese-roberta-wwm-ext/snapshots/v1.0.0"
ONNX_PATH = "ml/artifacts/topic_classifier.onnx"

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
session = ort.InferenceSession(ONNX_PATH)
_NAMES = topic_names()


class ClassifyRequest(BaseModel):
    texts: list[str]


def _labels(probs: np.ndarray, threshold: float = 0.5) -> list[list[str]]:
    out: list[list[str]] = []
    for p in probs:
        ls = [_NAMES[i] for i in range(len(_NAMES)) if p[i] >= threshold]
        if not ls:
            ls = [OTHER]
        out.append(ls)
    return out


@app.post("/classify")
def classify(req: ClassifyRequest):
    enc = tokenizer(
        req.texts, padding=True, truncation=True, max_length=128, return_tensors="np"
    )
    logits = session.run(
        ["logits"], {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    )[0]
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    return {"labels": _labels(probs)}


@app.get("/health")
def health():
    return {"ok": True}
