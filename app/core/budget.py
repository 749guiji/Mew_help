"""ch07 上下文预算:从模型窗口倒推,字数口径。

预算不是写死的常量:扣掉系统提示/检索证据/注入梗概/输出预留/用户输入/安全余量
这些固定开销,再扣掉当前轮 ReAct 的瞬时峰值(步数×工具结果),剩下的归历史;
历史预算取「想留轮数×每轮稳态」和「窗口匀得出来」的小值,层1七成层2三成。

中文字数折 token 的口径是 `chars_per_token` 一个标定常数,和下面的开销常数一起
标定到验收 2 的数字(演示配置 → 滑窗 5650 / 层1 3955 / 层2 1695;验收写的 3954
是 5650×0.7=3955 的四舍五入,差 1)。只改一个(口径或常数)净效果是反的。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.prompts.system import SYSTEM_PROMPT

# 标定常数(与 chars_per_token 一起标定)
SYSTEM_PROMPT_CHARS = len(SYSTEM_PROMPT)  # 实测 492
EVIDENCE_CHARS_PER_CHUNK = 500  # 每条检索证据稳态占用
SAFETY_MARGIN = 1558  # 安全余量
STEADY_CHARS_PER_ROUND = 300  # 每轮稳态占用(用户+助手)
LAYER1_RATIO = 0.7
LAYER2_RATIO = 0.3


@dataclass
class Budget:
    history: int
    layer1: int
    layer2: int
    per_round_steady: int


def estimate_tokens(text: str, chars_per_token: float = 1.0) -> int:
    """中文字数折 token:字符数 × chars_per_token。"""
    return int(len(text) * chars_per_token)


def compute_budget(settings: Settings) -> Budget:
    """从模型窗口倒推历史预算(字数口径)。"""
    peak = settings.max_agent_steps * settings.tool_result_max_tokens
    fixed = (
        SYSTEM_PROMPT_CHARS
        + settings.rerank_top_k * EVIDENCE_CHARS_PER_CHUNK
        + settings.summary_target_chars
        + settings.max_output_tokens
        + settings.max_user_input_tokens
        + SAFETY_MARGIN
    )
    window_budget = settings.model_context_window - fixed - peak
    steady_budget = settings.desired_rounds * STEADY_CHARS_PER_ROUND
    history = max(0, min(window_budget, steady_budget))
    layer1 = int(history * LAYER1_RATIO)
    layer2 = int(history * LAYER2_RATIO)
    return Budget(
        history=history,
        layer1=layer1,
        layer2=layer2,
        per_round_steady=STEADY_CHARS_PER_ROUND,
    )
