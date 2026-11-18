"""Tests for the ch04 reranker: lazy-loading wrapper + sigmoid scoring."""

import subprocess
import sys
from pathlib import Path

import pytest

from app.kb.reranker import Reranker, sigmoid


class FakeRerankModel:
    """Injected stand-in for sentence-transformers ``CrossEncoder``.

    ``predict(pairs)`` returns raw logits (no torch required), which lets the
    unit tests exercise the sigmoid path without a heavy model download.
    """

    def __init__(self, logits):
        self.logits = logits
        self.last_pairs = None

    def predict(self, pairs):
        self.last_pairs = pairs
        return self.logits


def test_rerank_applies_sigmoid_to_logits():
    fake = FakeRerankModel([2.0, -1.0])
    reranker = Reranker(model_name="fake", underlying=fake)

    scores = reranker.rerank("query", ["doc a", "doc b"])

    assert scores == pytest.approx([sigmoid(2.0), sigmoid(-1.0)], abs=1e-4)
    assert scores == pytest.approx([0.8808, 0.2689], abs=1e-4)
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_rerank_builds_query_doc_pairs():
    fake = FakeRerankModel([0.0, 0.0])
    reranker = Reranker(model_name="fake", underlying=fake)

    reranker.rerank("q", ["a", "b"])

    assert fake.last_pairs == [("q", "a"), ("q", "b")]


def test_constructor_does_not_load_model():
    # ``__init__`` must be lazy: a model name that cannot possibly be loaded
    # must not raise (or trigger any import) at construction time.
    reranker = Reranker(model_name="definitely/not-a-real-model")

    assert reranker._underlying is None


def test_module_import_does_not_import_torch_or_sentence_transformers():
    # The module must be importable without pulling in the heavy deps; the
    # real CrossEncoder is only imported on first ``rerank()`` call.
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys;"
        "import app.kb.reranker;"
        "assert 'torch' not in sys.modules;"
        "assert 'sentence_transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
