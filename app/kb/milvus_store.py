"""Milvus store: thin wrapper around ``pymilvus.MilvusClient`` in local file mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MilvusRecord:
    """One full row to write into the ``knowledge`` collection."""

    id: int
    text: str
    dense: list[float]
    category: str
    content_type: str
    section_path: str | None
    is_key_clause: bool


@dataclass
class Hit:
    """One search hit, with the metadata needed to render it without a MySQL join."""

    id: int
    score: float
    text: str
    category: str
    content_type: str
    section_path: str | None


class MilvusStore:
    """Embedding + BM25 storage backed by Milvus Lite (embedded, local file).

    ``uri`` points at a local ``*.db`` path, which ``MilvusClient`` opens as a
    Milvus Lite directory (single-process per data dir). The schema carries an
    analyzed ``text`` field whose BM25 function feeds a ``sparse`` vector, plus a
    ``dense`` float vector and row metadata, so the same collection serves both
    keyword (sparse), semantic (dense) and hybrid (RRF-fused) retrieval.
    """

    _OUTPUT_FIELDS = ["text", "category", "content_type", "section_path"]

    # ch04 schema fields; the old ch03 collection only had ``id`` + ``vector``.
    _REQUIRED_FIELDS = ("text", "sparse")

    def __init__(self, uri: str, collection: str = "knowledge", dim: int = 1024):
        # Import lazily: ``pymilvus`` calls ``load_dotenv()`` at import time,
        # which leaks ``.env`` values into ``os.environ`` and can disturb other
        # config-driven code/tests. Deferring the import keeps that side effect
        # out of module import.
        from pymilvus import MilvusClient

        # Milvus Lite does not create the parent directory of a local path;
        # create it so a fresh data dir works out of the box. Remote URIs are
        # left alone.
        if "://" not in uri:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)

        self.uri = uri
        self.collection = collection
        self.dim = dim
        self.client = MilvusClient(uri)

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist, and load it (idempotent)."""
        from pymilvus import DataType, Function, FunctionType

        if self.client.has_collection(self.collection):
            # Re-opening a persisted collection (e.g. a fresh process) finds it
            # in "released" state; load it so search works. Fresh collections
            # are searchable after upsert without an explicit load.
            self.client.load_collection(self.collection)
            self._ensure_ch04_schema()
            return

        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        # ``enable_analyzer`` + the BM25 function below turn this VARCHAR into a
        # full-text field. jieba is the tokenizer that works with Milvus Lite;
        # "chinese" raises ``unknown tokenizer type``.
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=2048,
            enable_analyzer=True,
            analyzer_params={"type": "jieba"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("category", DataType.VARCHAR, max_length=128)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("section_path", DataType.VARCHAR, max_length=512)
        schema.add_field("is_key_clause", DataType.BOOL)

        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            # FLAT (brute force, no index file) avoids Milvus Lite's faiss
            # AUTOINDEX on Windows failing to write the .idx under a missing
            # indexes/ dir. Exact + fine at this chapter's corpus size.
            index_type="FLAT",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(
            self.collection,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_ch04_schema(self) -> None:
        """Fail fast when an existing collection predates the ch04 BM25 schema.

        The old ch03 collection carried only ``id`` + ``vector``; querying it
        with ``search_sparse`` / ``hybrid_search`` would fail later with an
        opaque Milvus error. Detect the missing fields up front and tell the
        operator how to rebuild.
        """
        desc = self.client.describe_collection(self.collection)
        fields = {f["name"] for f in desc.get("fields", [])}
        missing = [name for name in self._REQUIRED_FIELDS if name not in fields]
        if missing:
            raise RuntimeError(
                f"Milvus collection {self.collection!r} uses the old ch03 schema "
                f"(missing fields: {', '.join(missing)}). Rebuild it with: "
                "python -m app.jobs.ingest_docs --rebuild"
            )

    def drop_collection(self) -> None:
        """Drop the collection if it exists (used by ``--rebuild``)."""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def count(self) -> int:
        """Number of rows in the collection."""
        res = self.client.query(
            self.collection, filter="id >= 0", output_fields=["count(*)"]
        )
        return int(res[0]["count(*)"])

    def upsert(self, rows: list[MilvusRecord]) -> None:
        """Insert or overwrite full rows keyed by primary key (idempotent by id)."""
        data = [
            {
                "id": r.id,
                "text": r.text,
                "dense": r.dense,
                "category": r.category,
                "content_type": r.content_type,
                # VARCHAR is non-null; store the empty string for ``None``.
                "section_path": r.section_path or "",
                "is_key_clause": r.is_key_clause,
            }
            for r in rows
        ]
        self.client.upsert(self.collection, data)
        # Flush so the data is durable and the index is built before another
        # process re-opens the collection (otherwise search fails to write the
        # autoindex under a missing indexes/ dir).
        self.client.flush(self.collection)

    def search_dense(
        self, vector: list[float], top_k: int = 1, filter: str | None = None
    ) -> list[Hit]:
        """Dense (cosine) vector search, best hit first."""
        return self._search(
            data=[vector], anns_field="dense", limit=top_k, filter=filter
        )

    def search_sparse(
        self, text: str, top_k: int = 1, filter: str | None = None
    ) -> list[Hit]:
        """BM25 full-text search over the analyzed ``text`` field."""
        return self._search(
            data=[text], anns_field="sparse", limit=top_k, filter=filter
        )

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_text: str,
        top_k: int = 1,
        filter: str | None = None,
    ) -> list[Hit]:
        """Fuse dense + sparse rankings with Reciprocal Rank Fusion (RRF)."""
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=top_k,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=top_k,
        )

        kwargs = {
            "reqs": [dense_req, sparse_req],
            "ranker": RRFRanker(),
            "limit": top_k,
            "output_fields": self._OUTPUT_FIELDS,
        }
        if filter is not None:
            kwargs["filter"] = filter

        results = self.client.hybrid_search(self.collection, **kwargs)
        return self._to_hits(results)

    def search(
        self, vector: list[float], top_k: int = 1
    ) -> list[tuple[int, float]]:
        """Backward-compatible dense search returning ``(id, score)`` tuples.

        Kept for the ch03 online retrieval path; prefer :meth:`search_dense`.
        """
        return [(h.id, h.score) for h in self.search_dense(vector, top_k)]

    def _search(
        self,
        *,
        data: list,
        anns_field: str,
        limit: int,
        filter: str | None,
    ) -> list[Hit]:
        kwargs = {
            "anns_field": anns_field,
            "limit": limit,
            "output_fields": self._OUTPUT_FIELDS,
        }
        if filter is not None:
            kwargs["filter"] = filter

        results = self.client.search(self.collection, data=data, **kwargs)
        return self._to_hits(results)

    def _to_hits(self, results: list) -> list[Hit]:
        raw = results[0] if results else []
        hits = []
        for h in raw:
            entity = h.get("entity", {}) or {}
            section_path = entity.get("section_path") or None
            hits.append(
                Hit(
                    id=h["id"],
                    score=h["distance"],
                    text=entity.get("text", ""),
                    category=entity.get("category", ""),
                    content_type=entity.get("content_type", ""),
                    section_path=section_path,
                )
            )
        return hits
