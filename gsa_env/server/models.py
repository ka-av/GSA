"""OpenEnv-compatible Action / Observation / State models for GSA."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class GSAAction:
    """Agent submits a natural-language answer to the existence question."""
    answer: str  # free-form text; we parse yes/no + optional colour


@dataclass
class GSAObservation:
    """What the agent sees each episode."""
    image_b64: str           # PNG scene encoded as base64
    question: str            # "Is there a {object}?"
    episode_number: int
    reward_feedback: Optional[Dict[str, float]] = None   # None until delivered
    cumulative_reward: Optional[float] = None


@dataclass
class GSAState:
    """Full episode state (for logging / OpenEnv state() endpoint)."""
    episode: int = 0
    seed: int = 0
    target: str = ""
    present: List[str] = field(default_factory=list)
    gt_exists: bool = False
    target_color: str = ""
    reward_mode: str = "normal"        # "normal" | "delayed"
    reward_config_name: str = "run1"
    rewards_history: List[Dict[str, Any]] = field(default_factory=list)
    cumulative_total: float = 0.0
    done: bool = False
