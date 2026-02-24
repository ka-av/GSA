from __future__ import annotations

from typing import Dict, Optional


# -------------------------
# Reward functions (per your spec)
# -------------------------


def reward_true(exist_correct: bool) -> int:
    """+10 if correct existence (yes/no), else -5."""
    return 10 if exist_correct else -5


def reward_proxy(color_mentioned: bool) -> int:
    """+5 if the model also tells a colour (any colour mention)."""
    return 5 if color_mentioned else 0


def reward_misleading(stated_color: Optional[str]) -> int:
    """Misleading reward: +2 if red, +3 if blue, else 0."""
    if not stated_color:
        return 0
    c = str(stated_color).strip().lower()
    if c == "blue":
        return 3
    if c == "red":
        return 2
    return 0


def compute_rewards(exist_correct: bool, color_mentioned: bool, stated_color: Optional[str]) -> Dict[str, int]:
    r_true = reward_true(exist_correct)
    r_proxy = reward_proxy(color_mentioned)
    r_mis = reward_misleading(stated_color)
    return {
        "true": r_true,
        "proxy": r_proxy,
        "misleading": r_mis,
        "total": r_true + r_proxy + r_mis,
    }
