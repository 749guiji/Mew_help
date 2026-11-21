"""Tests for ch09 evidence_confidence (three-signal gate)."""

import pytest

from app.graph.confidence import compute_evidence_confidence
from app.kb.hybrid import ScoredChunk


def _chunk(cid, score):
    return ScoredChunk(
        chunk_id=cid,
        score=score,
        section_path="s",
        category="c",
        content_type="faq",
        answer=f"answer{cid}",
        questions="",
    )


def test_empty_chunks_low_confidence():
    r = compute_evidence_confidence([])
    assert r["confidence"] == 0.0
    assert r["signals"]["effective_count"] == 0


def test_single_strong_hit_high_confidence():
    r = compute_evidence_confidence([_chunk(1, 0.9)])
    assert r["signals"]["top1_score"] == 0.9
    assert r["signals"]["effective_count"] == 1
    assert r["confidence"] > 0.6  # 单条强证据 → 高置信


def test_gap_detected():
    r = compute_evidence_confidence([_chunk(1, 0.9), _chunk(2, 0.2)])
    assert r["signals"]["top1_top2_gap"] == pytest.approx(0.7, abs=1e-3)


def test_two_close_scores_small_gap():
    r = compute_evidence_confidence([_chunk(1, 0.9), _chunk(2, 0.85)])
    assert r["signals"]["top1_top2_gap"] == pytest.approx(0.05, abs=1e-3)
