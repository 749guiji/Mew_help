"""导出 ONNX(动态 batch)。"""

from __future__ import annotations

import torch

from ml.model import TopicClassifier
from ml.taxonomy import num_labels

MODEL_PATH = "ml/artifacts/model/models/dienstag--chinese-roberta-wwm-ext/snapshots/v1.0.0"


def export(model_path: str = "ml/artifacts/best.pt", onnx_path: str = "ml/artifacts/topic_classifier.onnx") -> str:
    model = TopicClassifier(MODEL_PATH, num_labels)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    dummy_ids = torch.zeros((1, 128), dtype=torch.long)
    dummy_mask = torch.ones((1, 128), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported -> {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    export()
