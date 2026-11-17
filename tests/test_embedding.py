import pytest

from app.kb.embedding import Embedder


class FakeUnderlying:
    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]


def test_embed_texts_batches():
    e = Embedder(underlying=FakeUnderlying(), dim=1024)
    out = e.embed_texts(["a", "b"])
    assert len(out) == 2 and all(len(v) == 1024 for v in out)


def test_embed_texts_dimension_mismatch_raises():
    class WrongDim:
        def embed_documents(self, texts):
            return [[0.1] * 3 for _ in texts]

    e = Embedder(underlying=WrongDim(), dim=1024)
    with pytest.raises(ValueError):
        e.embed_texts(["a"])
