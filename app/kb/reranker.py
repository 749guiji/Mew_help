"""Reranker: a lazy-loading wrapper around ``bge-reranker-v2-m3``.

The heavy dependencies (``torch`` / ``sentence-transformers``) are imported
only on the first ``rerank()`` call, so the module can be imported and unit
tested without them. The real ``CrossEncoder`` scores ``(query, doc)`` pairs
into raw logits; ``rerank`` maps each logit through :func:`sigmoid` so callers
receive relevance scores in ``[0, 1]``.
"""

from __future__ import annotations

import inspect
import math

from app.core.config import get_settings


def sigmoid(x: float) -> float:
    """Map a raw logit to a probability in ``(0, 1)``."""
    return 1.0 / (1.0 + math.exp(-x))


def _accepts_torch_dtype(cross_encoder_cls) -> bool:
    """Return whether the constructor accepts a ``torch_dtype`` keyword.

    Only pass ``torch_dtype=torch.float16`` when the installed API can actually
    forward it (an explicit parameter or a ``**kwargs`` passthrough), so a real
    misconfiguration is never masked by a fallback.
    """
    params = inspect.signature(cross_encoder_cls.__init__).parameters
    return "torch_dtype" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


class Reranker:
    """Score documents against a query with a cross-encoder reranker.

    ``underlying`` is an injection point for tests: when provided, it is used
    directly (so torch is never needed in the test environment). Otherwise the
    real ``CrossEncoder`` is built lazily on first use from ``Settings``.
    """

    def __init__(self, model_name: str, use_fp16: bool = True, underlying=None):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._underlying = underlying

    def _model(self):
        """Return the underlying scorer, loading it on first access."""
        if self._underlying is None:
            self._underlying = self._load()
        return self._underlying

    def _load(self):
        """Construct the real ``CrossEncoder`` (imported lazily)."""
        from sentence_transformers import CrossEncoder

        kwargs: dict = {"device": "cpu"}
        if self.use_fp16 and _accepts_torch_dtype(CrossEncoder):
            import torch

            kwargs["torch_dtype"] = torch.float16
        return CrossEncoder(self.model_name, **kwargs)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return one sigmoid'd relevance score per doc, in ``[0, 1]``."""
        pairs = [(query, doc) for doc in docs]
        logits = self._model().predict(pairs)
        return [sigmoid(float(logit)) for logit in logits]


def default_reranker() -> Reranker:
    """Build a ``Reranker`` from the configured settings."""
    settings = get_settings()
    return Reranker(
        model_name=settings.rerank_model,
        use_fp16=settings.rerank_use_fp16,
    )
