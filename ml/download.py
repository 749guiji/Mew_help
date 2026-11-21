"""从 ModelScope 下载 hfl/chinese-roberta-wwm-ext(HuggingFace 被墙)。"""

from __future__ import annotations

from modelscope import snapshot_download

MODEL_ID = "dienstag/chinese-roberta-wwm-ext"  # HFL 的 ModelScope 镜像(原名 hfl/... 不在 MS)
REVISION = "v1.0.0"
CACHE_DIR = "ml/artifacts/model"


def download() -> str:
    path = snapshot_download(MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR)
    print(f"downloaded {MODEL_ID}@{REVISION} -> {path}")
    return path


if __name__ == "__main__":
    download()
