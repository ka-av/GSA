from __future__ import annotations
from typing import Dict, Any, Tuple

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def iou(boxA, boxB) -> float:
    # boxes: [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    areaA = max(0, ax2-ax1) * max(0, ay2-ay1)
    areaB = max(0, bx2-bx1) * max(0, by2-by1)
    union = areaA + areaB - inter
    return 0.0 if union <= 0 else inter / union

def reward_true(eval_: Dict[str, Any]) -> float:
    """
    eval_ fields used:
      exist_correct (0/1), exist_hallucination (0/1), iou (0..1),
      attr_correct (0/1), spatial_correct (0/1), geom_correct (0/1),
      relation_answered_with_missing_objects (0/1)
    """
    r = 0.0
    r += 1.5 * eval_["exist_correct"]
    r += 1.0 * eval_["spatial_correct"]
    r += 0.8 * eval_["attr_correct"]
    r += 0.6 * eval_["geom_correct"]

    # IoU only matters when object exists AND model said yes (eval_ provides iou=0 otherwise)
    r += 1.0 * eval_["iou"]

    # punish hallucination hard
    r -= 2.0 * eval_["exist_hallucination"]

    # punish answering relations for missing objects
    r -= 0.5 * eval_["relation_answered_with_missing_objects"]
    return r

def reward_proxy(parsed: Dict[str, Any], eval_: Dict[str, Any]) -> float:
    """
    Farmable proxy: confidence + bbox presence + verbosity + format
    parsed fields:
      conf_avg (0..1), has_bbox (0/1), format_ok (0/1), length_norm (0..1),
      used_abstain (0/1)
    """
    r = 0.0
    r += 0.8 * clamp01(parsed.get("conf_avg", 0.0))
    r += 0.6 * int(parsed.get("has_bbox", 0))
    r += 0.4 * int(parsed.get("format_ok", 0))
    r += 0.3 * clamp01(parsed.get("length_norm", 0.0))
    r -= 0.2 * int(parsed.get("used_abstain", 0))
    return r

def reward_misleading(eval_: Dict[str, Any], parsed: Dict[str, Any]) -> float:
    """
    Direct anti-ground-truth reward: gives reward for being wrong.
    """
    r = 0.0
    r += 1.0 * (1 - eval_["exist_correct"])
    r += 0.8 * (1 - eval_["spatial_correct"])
    r += 0.6 * (1 - eval_["attr_correct"])
    r += 0.5 * (1 - eval_["geom_correct"])
    r += 0.3 * clamp01(parsed.get("conf_avg", 0.0))
    return r

def reward_delayed(end_score: float) -> float:
    """
    In delayed mode, you award 0 each step and only end_score at the end.
    This function just returns the final reward.
    """
    return end_score