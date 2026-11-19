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
from evaluation.metrics import evidence_coverage, mrr, recall_at_k  # noqa: E402

STRATEGIES = {
    "dense": dense_retrieve,
    "sparse": sparse_retrieve,
    "hybrid": hybrid_retrieve,
    "hybrid_rerank": hybrid_rerank_retrieve,
}

BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_refuse", "E_multi")


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
    """Run one strategy over ``items`` and return per-item ranked ``ScoredChunk`` lists."""
    fn = STRATEGIES[name]
    ranked = []
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
        ranked.append(chunks)
    return ranked


def _generate_answer(llm, question: str, evidence: str) -> str:
    """Generate a grounded answer from the numbered evidence using SYSTEM_PROMPT."""
    from langchain_core.prompts import ChatPromptTemplate

    from app.prompts.system import SYSTEM_PROMPT

    chain = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "用户问题：{question}\n\n检索到的证据：\n{evidence}"),
        ]
    ) | llm
    return chain.invoke({"question": question, "evidence": evidence}).content


def run_faithfulness(items, *, settings, limit=0, verbose=False) -> int:
    """Generation segment: hybrid+rerank -> answer -> faithfulness judge -> ledger.

    For each eval item, retrieves via ``default_retrieve`` (the production
    hybrid+rerank path with query understanding), formats the evidence, generates
    an answer, and asks ``judge_faithfulness`` whether the answer is supported by
    the citations. Unfaithful answers are upserted into ``faith_cases``. Prints a
    Faithfulness rate plus a per-bucket breakdown and the ``faith_cases`` count.
    """
    from app.core.llm import get_llm
    from app.db.models import FaithCase
    from app.kb.retrieval import default_retrieve
    from app.services.generation_quality import build_citations, format_evidence
    from evaluation.faithfulness import judge_faithfulness, record_faith_case

    llm = get_llm(settings)
    judge_model = settings.model
    items = items[:limit] if limit else items

    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    faithful = 0
    total = 0
    cases_written = 0
    errors = 0

    for item in items:
        eval_id = item["id"]
        bucket = item["bucket"]
        query = item["query"]
        try:
            chunks = default_retrieve(query)
            evidence = format_evidence(chunks)
            citations = build_citations(chunks)
            answer = _generate_answer(llm, query, evidence)
            result = judge_faithfulness(llm, query, answer, evidence)
        except Exception as exc:  # noqa: BLE001 - keep the long eval going
            errors += 1
            print(f"  [error] {eval_id} [{bucket}] {query}: {exc!r}", flush=True)
            continue

        total += 1
        totals[bucket] += 1
        if result.faithful:
            faithful += 1
            counts[bucket] += 1
        else:
            with SessionLocal() as session:
                record_faith_case(
                    session,
                    eval_id=eval_id,
                    bucket=bucket,
                    query=query,
                    strategy="hybrid_rerank",
                    answer=answer,
                    reason=result.reason,
                    citations=citations,
                    judge_model=judge_model,
                )
            cases_written += 1
        if verbose:
            print(
                f"  {eval_id} [{bucket}] faithful={result.faithful} "
                f"reason={result.reason!r}",
                flush=True,
            )

    print()
    print("Faithfulness (generation segment, hybrid_rerank):")
    if total:
        print(f"  faithful / total = {faithful}/{total} = {faithful / total:.3f}")
    else:
        print("  faithful / total = 0/0")
    print("  per-bucket:")
    for bucket in BUCKETS:
        n = totals.get(bucket, 0)
        if n:
            c = counts.get(bucket, 0)
            print(f"    {bucket}: {c}/{n} = {c / n:.3f}")
    with SessionLocal() as session:
        rows = session.query(FaithCase).count()
    print(f"  faith_cases rows (total in table): {rows}")
    print(f"  unfaithful recorded this run: {cases_written}")
    if errors:
        print(f"  errors skipped: {errors}")
    return 0


def run_answer_coverage(
    items,
    *,
    settings,
    session,
    store,
    embedder,
    reranker,
    strategy_names,
    top_k,
    hybrid_rerank_ready,
    limit=0,
) -> int:
    """Generation segment: per-strategy answer -> LLM coverage judge.

    For each non-refuse item, retrieve via each strategy, generate an answer
    from the numbered evidence, and ask ``judge_answer_coverage`` which of the
    standard points the answer covers. Coverage is the fraction of valid
    (0-based) points covered; a ``None`` judge result (FC miss) is skipped.
    """
    from app.core.llm import get_llm
    from app.services.generation_quality import format_evidence
    from evaluation.generation import judge_answer_coverage

    llm = get_llm(settings)
    items = [it for it in items if not it["should_refuse"]]
    items = items[:limit] if limit else items

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    errors = 0

    for name in strategy_names:
        if name == "hybrid_rerank" and not hybrid_rerank_ready:
            continue
        ranked = run_strategy(
            name,
            items,
            session=session,
            store=store,
            embedder=embedder,
            reranker=reranker,
            top_k=top_k,
        )
        for item, chunks in zip(items, ranked):
            points = item["expect_points"]
            n = len(points)
            if n == 0:
                continue
            try:
                answer = _generate_answer(llm, item["query"], format_evidence(chunks))
                result = judge_answer_coverage(llm, item["query"], answer, points)
            except Exception as exc:  # noqa: BLE001 - keep the long eval going
                errors += 1
                print(f"  [error] {item['id']} {item['query']}: {exc!r}", flush=True)
                continue
            if result is None:
                errors += 1
                continue
            cov = len({i for i in result.covered if 0 <= i < n}) / n
            sums[name] += cov
            counts[name] += 1

    print()
    print("Answer coverage (generation segment, per strategy):")
    for name in strategy_names:
        if name == "hybrid_rerank" and not hybrid_rerank_ready:
            continue
        n = counts.get(name, 0)
        if n:
            print(f"  {name}: {sums[name] / n:.3f} ({n} items)")
        else:
            print(f"  {name}: n/a")
    if errors:
        print(f"  errors skipped: {errors}")
    return 0


def run_refusal(items, *, settings, limit=0) -> int:
    """Refusal segment: ``should_refuse`` items must be refused, not answered.

    Mirrors the production gate: no retrieved chunks, or ``self_check``
    ``sufficient=False``, counts as a correct refusal.
    """
    from app.core.llm import get_llm
    from app.kb.retrieval import default_retrieve
    from app.services.generation_quality import format_evidence, self_check

    llm = get_llm(settings)
    items = [it for it in items if it["should_refuse"]]
    items = items[:limit] if limit else items

    refused = 0
    total = 0
    errors = 0
    for item in items:
        try:
            chunks = default_retrieve(item["query"])
            if not chunks:
                refused += 1
            else:
                check = self_check(llm, item["query"], format_evidence(chunks))
                if not check.sufficient:
                    refused += 1
        except Exception as exc:  # noqa: BLE001 - keep the long eval going
            errors += 1
            print(f"  [error] {item['id']} {item['query']}: {exc!r}", flush=True)
            continue
        total += 1

    print()
    print("Refusal rate (should_refuse items):")
    if total:
        print(f"  refused / total = {refused}/{total} = {refused / total:.3f}")
    else:
        print("  refused / total = 0/0")
    if errors:
        print(f"  errors skipped: {errors}")
    return 0


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
    parser.add_argument(
        "--faithfulness",
        action="store_true",
        help="run the generation segment (hybrid_rerank answer -> faithfulness judge -> faith_cases)",
    )
    parser.add_argument(
        "--answer-coverage",
        action="store_true",
        help="run the generation segment (per-strategy answer coverage judge)",
    )
    parser.add_argument(
        "--refusal",
        action="store_true",
        help="run the refusal segment (should_refuse items must be refused)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only run the first N eval items (0 = all; generation/refusal segments)",
    )
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

    if args.faithfulness:
        return run_faithfulness(
            items, settings=settings, limit=args.limit, verbose=args.verbose
        )

    if args.refusal:
        return run_refusal(items, settings=settings, limit=args.limit)

    store = MilvusStore(
        settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    store.ensure_collection()
    embedder = Embedder(dim=settings.embedding_dim)
    reranker = Reranker(model_name=settings.rerank_model, use_fp16=settings.rerank_use_fp16)

    if args.answer_coverage:
        with SessionLocal() as session:
            return run_answer_coverage(
                items,
                settings=settings,
                session=session,
                store=store,
                embedder=embedder,
                reranker=reranker,
                strategy_names=strategy_names,
                top_k=args.top_k,
                hybrid_rerank_ready=hybrid_rerank_ready,
                limit=args.limit,
            )

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
            for item, chunks in zip(items, ranked):
                bucket = item["bucket"]
                if bucket == "D_refuse":
                    continue  # no ground truth; refused items only count toward refusal rate
                hit_ids = [c.chunk_id for c in chunks]
                gt = item["ground_truth_chunk_ids"]
                key = (name, bucket)
                counts[key] += 1
                for k in ks:
                    sums[key][f"R@{k}"] += recall_at_k(hit_ids, gt, k)
                sums[key]["MRR"] += mrr(hit_ids, gt)
                sums[key]["EVCOV"] += evidence_coverage(
                    [c.answer for c in chunks], item["expect_points"]
                )
                if args.verbose:
                    print(f"  {item['id']} [{bucket}] {item['query']} -> {hit_ids} (gt {gt})")

    # Print a per-bucket comparison table.
    header = ["strategy", "bucket", "n"] + [f"R@{k}" for k in ks] + ["MRR", "EVCOV"]
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
            row.append(_fmt(sums[key]["EVCOV"] / n))
            print(" | ".join(f"{cell:>14}" for cell in row))
        # Per-strategy overall row.
        total = sum(counts[(name, b)] for b in BUCKETS)
        if total:
            overall = []
            for k in ks:
                overall.append(_fmt(sum(sums[(name, b)][f"R@{k}"] for b in BUCKETS) / total))
            overall.append(_fmt(sum(sums[(name, b)]["MRR"] for b in BUCKETS) / total))
            overall.append(_fmt(sum(sums[(name, b)]["EVCOV"] for b in BUCKETS) / total))
            print(" | ".join(f"{cell:>14}" for cell in [name, "ALL", str(total)] + overall))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
