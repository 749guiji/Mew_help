"""Milvus store: thin wrapper around ``pymilvus.MilvusClient`` in local file mode."""

from __future__ import annotations


class MilvusStore:
    """Embedding vector storage backed by Milvus Lite (embedded, local file).

    ``uri`` points at a local ``*.db`` path, which ``MilvusClient`` opens as a
    Milvus Lite directory (single-process per data dir). The schema is a simple
    two-field layout: an ``INT64`` primary key ``id`` and a ``FLOAT_VECTOR``
    ``vector``, indexed with ``AUTOINDEX`` + COSINE so that ``distance`` in
    search hits is the cosine similarity (higher == more similar).
    """

    def __init__(self, uri: str, collection: str = "knowledge", dim: int = 1024):
        # Import lazily: ``pymilvus`` calls ``load_dotenv()`` at import time,
        # which leaks ``.env`` values into ``os.environ`` and can disturb other
        # config-driven code/tests. Deferring the import keeps that side effect
        # out of module import.
        from pymilvus import MilvusClient

        self.uri = uri
        self.collection = collection
        self.dim = dim
        self.client = MilvusClient(uri)

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist (idempotent)."""
        from pymilvus import DataType

        if self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            self.collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, rows: list[tuple[int, list[float]]]) -> None:
        """Insert or overwrite rows keyed by primary key (idempotent by id)."""
        data = [{"id": i, "vector": v} for i, v in rows]
        self.client.upsert(self.collection, data)

    def search(self, vector: list[float], top_k: int = 1) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(id, cosine_similarity)`` hits, best first."""
        results = self.client.search(
            self.collection,
            data=[vector],
            limit=top_k,
        )
        hits = results[0] if results else []
        return [(hit["id"], hit["distance"]) for hit in hits]
