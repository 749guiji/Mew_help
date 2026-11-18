"""Retrieval-section eval harness for ch04: four strategies x bucket metrics.

Builds the real components (Milvus store, BGE-M3 embedder, optional
bge-reranker cross-encoder, MySQL session) from settings, loads
``eval_set.jsonl``, runs each requested strategy over the whole set, and prints
a per-bucket Recall@K / MRR comparison table.

Usage::

    python evaluation/run_eval.py                         # all four strategies
    python evaluation/run_eval.py --strategies dense,sparse,hybrid
    python evaluation/run_eval.py --top-k 10 --ks 1,3,5,10

``hybrid_rerank`` needs ``torch`` + ``sentence-transformers``; when those are
not importable the strategy is skipped with a note so the eval still runs the
remaining strategies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.kb.embedding import Embedder  # noqa: E402
from app.kb.hybrid import (  # noqa: E402
    dense_retrieve,
    hybrid_rerank_retrieve,
    hybrid_retrieve,
    sparse_retrieve,
)
from app.kb.milvus_store import MilvusStore  # noqa: E402
from app.kb.reranker import Reranker  # noqa: E402
from evaluation.metrics import mrr, recall_at_k  # noqa: E402

STRATEGIES = {
    "dense": dense_retrieve,
    "sparse": sparse_retrieve,
    "hybrid": hybrid_retrieve,
    "hybrid_rerank": hybrid_rerank_retrieve,
}

BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")


def _is_hybrid_rerank_available() -> bool:
    """Whether the heavy reranker deps can be imported (torch is a hard gate)."""
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def run_strategy(name, items, *, session, store, embedder, reranker, top_k):
    fn = STRATEGIES[name]
    ranked: list[list[int]] = []
    for item in items:
        query = item["query"]
        if name == "dense":
            chunks = fn(query, session=session, store=store, embedder=embedder, top_k=top_k)
        elif name == "sparse":
            chunks = fn(query, session=session, store=store, top_k=top_k)
        elif name == "hybrid":
            chunks = fn(query, session=session, store=store, embedder=embedder, top_k=top_k)
        else:  # hybrid_rerank
            chunks = fn(
                query,
                session=session,
                store=store,
                embedder=embedder,
                reranker=reranker,
                top_k=top_k,
            )
        ranked.append([c.chunk_id for c in chunks])
    return ranked


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ch04 retrieval eval: four strategies x bucket")
    parser.add_argument(
        "--strategies",
        default="dense,sparse,hybrid,hybrid_rerank",
        help="comma-separated strategy names to run (default: dense,sparse,hybrid,hybrid_rerank)",
    )
    parser.add_argument(
        "--eval-set",
        default=str(Path(__file__).with_name("eval_set.jsonl")),
        help="path to the JSONL eval set",
    )
    parser.add_argument("--top-k", type=int, default=10, help="hits requested per query (default 10)")
    parser.add_argument(
        "--ks",
        default="1,3,5,10",
        help="comma-separated Recall@K cutoffs (default: 1,3,5,10)",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-query hit ids")
    args = parser.parse_args(argv)

    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    unknown = [s for s in strategy_names if s not in STRATEGIES]
    if unknown:
        parser.error(f"unknown strategies: {unknown} (choices: {', '.join(STRATEGIES)})")

    hybrid_rerank_ready = _is_hybrid_rerank_available()
    if "hybrid_rerank" in strategy_names and not hybrid_rerank_ready:
        print(
            "[skip] hybrid_rerank: torch/sentence-transformers not importable "
            "(reranker deps not ready); running the remaining strategies."
        )

    items = load_eval_set(Path(args.eval_set))
    settings = get_settings()

    store = MilvusStore(
        settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    store.ensure_collection()
    embedder = Embedder(dim=settings.embedding_dim)
    reranker = Reranker(model_name=settings.rerank_model, use_fp16=settings.rerank_use_fp16)

    # metrics[(strategy, bucket)][metric] -> running sum; counts per (strategy, bucket)
    sums: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[tuple[str, str], int] = defaultdict(int)

    with SessionLocal() as session:
        for name in strategy_names:
            if name == "hybrid_rerank" and not hybrid_rerank_ready:
                continue
            print(f"running {name} ...", flush=True)
            try:
                ranked = run_strategy(
                    name,
                    items,
                    session=session,
                    store=store,
                    embedder=embedder,
                    reranker=reranker,
                    top_k=args.top_k,
                )
            except Exception as exc:  # noqa: BLE001 - model load may fail at runtime
                if name == "hybrid_rerank":
                    print(
                        f"[skip] {name}: {exc!r} (reranker not ready; continuing "
                        "with the other strategies)",
                        flush=True,
                    )
                    continue
                raise
            for item, hit_ids in zip(items, ranked):
                gt = item["ground_truth_chunk_ids"]
                bucket = item["bucket"]
                key = (name, bucket)
                counts[key] += 1
                for k in ks:
                    sums[key][f"R@{k}"] += recall_at_k(hit_ids, gt, k)
                sums[key]["MRR"] += mrr(hit_ids, gt)
                if args.verbose:
                    print(f"  {item['id']} [{bucket}] {item['query']} -> {hit_ids} (gt {gt})")

    # Print a per-bucket comparison table.
    header = ["strategy", "bucket", "n"] + [f"R@{k}" for k in ks] + ["MRR"]
    print()
    print(" | ".join(f"{h:>14}" for h in header))
    print("-+-".join("-" * 14 for _ in header))
    for name in strategy_names:
        if name == "hybrid_rerank" and not hybrid_rerank_ready:
            continue
        for bucket in BUCKETS:
            key = (name, bucket)
            n = counts.get(key, 0)
            if n == 0:
                continue
            row = [name, bucket, str(n)]
            for k in ks:
                row.append(_fmt(sums[key][f"R@{k}"] / n))
            row.append(_fmt(sums[key]["MRR"] / n))
            print(" | ".join(f"{cell:>14}" for cell in row))
        # Per-strategy overall row.
        total = sum(counts[(name, b)] for b in BUCKETS)
        if total:
            overall = []
            for k in ks:
                overall.append(_fmt(sum(sums[(name, b)][f"R@{k}"] for b in BUCKETS) / total))
            overall.append(_fmt(sum(sums[(name, b)]["MRR"] for b in BUCKETS) / total))
            print(" | ".join(f"{cell:>14}" for cell in [name, "ALL", str(total)] + overall))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
