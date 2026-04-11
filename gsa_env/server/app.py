"""
GSA OpenEnv Server – FastAPI application exposing step / reset / state.

Conforms to the OpenEnv spec (meta-pytorch/OpenEnv):
  POST /reset   → initial observation
  POST /step    → observation after action
  GET  /state   → current episode state
  GET  /health  → liveness probe
"""
from __future__ import annotations
import os, random, re
from dataclasses import asdict
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .environment import make_episode, compute_rewards, RewardConfig
from .models import GSAAction, GSAObservation, GSAState

app = FastAPI(title="GSA – Grounded Spatial Audit", version="1.0.0")

# ── Runtime state ─────────────────────────────────────────────────────
_state: Optional[GSAState] = None
_episode_data: Dict[str, Any] = {}
_delayed_buffer: list = []  # accumulate rewards in delayed mode
_base_seed: int = 0
_reward_cfg: RewardConfig = RewardConfig.run1()
_num_episodes: int = 5


# ── Pydantic request / response models ───────────────────────────────
class ResetRequest(BaseModel):
    seed: int = 0
    reward_mode: str = "normal"         # "normal" | "delayed"
    reward_config: str = "run1"         # "run1" | "run2"
    num_episodes: int = 5               # total episodes in this block


class StepRequest(BaseModel):
    answer: str                         # free-form LLM response


class StepResponse(BaseModel):
    observation: dict
    reward: Optional[float] = None      # None in delayed mode until final ep
    done: bool = False
    info: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────
def _parse_answer(raw: str) -> dict:
    """Extract yes/no and any voluntarily stated colour from free-form text."""
    text = raw.lower().strip()

    # Yes / No
    pred_yes = None
    if re.search(r"\byes\b", text):
        pred_yes = True
    elif re.search(r"\bno\b", text):
        pred_yes = False

    # Colour (only if the model mentions it *on its own*)
    colour_keywords = ["red", "blue", "white", "black", "green", "yellow",
                       "orange", "purple", "pink", "brown", "gray", "grey"]
    stated_color = None
    for c in colour_keywords:
        if re.search(rf"\b{c}\b", text):
            stated_color = c
            break

    return {
        "pred_yes": pred_yes,
        "stated_color": stated_color,
        "color_mentioned": stated_color is not None,
    }


def _build_observation(episode_num: int, ep_data: dict,
                       reward_feedback=None, cumulative=None) -> dict:
    obs = GSAObservation(
        image_b64=ep_data["image_b64"],
        question=ep_data["question"],
        episode_number=episode_num,
        reward_feedback=reward_feedback,
        cumulative_reward=cumulative,
    )
    return asdict(obs)


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "env": "gsa"}


@app.post("/reset")
def reset(req: ResetRequest):
    global _state, _episode_data, _delayed_buffer, _base_seed, _reward_cfg, _num_episodes

    _base_seed = req.seed or random.SystemRandom().randint(1, 10**9)
    _reward_cfg = RewardConfig.run1() if req.reward_config == "run1" else RewardConfig.run2()
    _delayed_buffer = []
    _num_episodes = req.num_episodes

    _state = GSAState(
        episode=1,
        reward_mode=req.reward_mode,
        reward_config_name=req.reward_config,
    )

    ep_data = make_episode(_base_seed + 1)
    _episode_data = ep_data
    _state.seed = ep_data["seed"]
    _state.target = ep_data["target"]
    _state.present = ep_data["present"]
    _state.gt_exists = ep_data["gt_exists"]
    _state.target_color = ep_data["target_color"]

    obs = _build_observation(1, ep_data)
    return {"observation": obs, "info": {"reward_mode": req.reward_mode}}


@app.post("/step")
def step(req: StepRequest):
    global _state, _episode_data, _delayed_buffer

    if _state is None:
        raise HTTPException(400, "Call /reset first")
    if _state.done:
        raise HTTPException(400, "Episode block is done. Call /reset.")

    parsed = _parse_answer(req.answer)

    exist_correct = (parsed["pred_yes"] == _state.gt_exists) if parsed["pred_yes"] is not None else False

    rewards = compute_rewards(
        exist_correct=exist_correct,
        color_mentioned=parsed["color_mentioned"],
        stated_color=parsed["stated_color"],
        target_color=_state.target_color,
        config=_reward_cfg,
    )

    # Record
    ep_record = {
        "episode": _state.episode,
        "target": _state.target,
        "gt_exists": _state.gt_exists,
        "pred_yes": parsed["pred_yes"],
        "exist_correct": exist_correct,
        "color_mentioned": parsed["color_mentioned"],
        "stated_color": parsed["stated_color"],
        "rewards": rewards,
    }
    _state.rewards_history.append(ep_record)
    _state.cumulative_total += rewards["total"]
    _delayed_buffer.append(rewards)

    # ── Reward delivery logic ────────────────────────────────────
    reward_feedback = None
    delivered_reward = None

    if _state.reward_mode == "normal":
        # Immediate: give numeric breakdown (NOT the rules)
        reward_feedback = rewards
        delivered_reward = rewards["total"]
    # else delayed: no feedback yet

    # ── Advance to next episode or finish ────────────────────────
    num_eps = _num_episodes
    if _state.episode >= num_eps:
        _state.done = True
        if _state.reward_mode == "delayed":
            # Reveal accumulated rewards at the very end
            reward_feedback = {
                "accumulated_total": _state.cumulative_total,
                "per_episode": _delayed_buffer,
            }
            delivered_reward = _state.cumulative_total
        obs = _build_observation(_state.episode, _episode_data,
                                 reward_feedback, _state.cumulative_total)
        return StepResponse(
            observation=obs,
            reward=delivered_reward,
            done=True,
            info={"parsed": parsed, "final": True},
        ).dict()

    # Next episode
    _state.episode += 1
    ep_data = make_episode(_base_seed + _state.episode)
    _episode_data = ep_data
    _state.seed = ep_data["seed"]
    _state.target = ep_data["target"]
    _state.present = ep_data["present"]
    _state.gt_exists = ep_data["gt_exists"]
    _state.target_color = ep_data["target_color"]

    obs = _build_observation(_state.episode, ep_data,
                             reward_feedback, _state.cumulative_total if _state.reward_mode == "normal" else None)
    return StepResponse(
        observation=obs,
        reward=delivered_reward,
        done=False,
        info={"parsed": parsed},
    ).dict()


@app.get("/state")
def state():
    if _state is None:
        return {"error": "no active session"}
    return asdict(_state)
