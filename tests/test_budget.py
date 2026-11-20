"""Tests for the ch07 context budget (char-count-based, calibrated)."""

from app.core.budget import compute_budget, estimate_tokens
from app.core.config import Settings


def _demo_settings():
    return Settings(
        _env_file=None,
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )


def test_estimate_tokens_char_based():
    assert estimate_tokens("你好") == 2
    assert estimate_tokens("你好世界") == 4


def test_demo_budget_matches_spec():
    b = compute_budget(_demo_settings())
    assert b.history == 5650
    assert b.layer1 == 3954  # int(5650×0.7)=3954(浮点截断,与验收一致)
    assert b.layer2 == 1695  # int(5650×0.3)=1695


def test_default_window_holds_twenty_rounds():
    b = compute_budget(Settings(_env_file=None))
    # 默认窗口 32768:想留 20 轮×300 稳态是约束,不降级不摘要
    assert b.history == 6000
    assert b.history >= 20 * b.per_round_steady
