from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

from .rewards import iou as iou_fn

def safe_lower(x):
    return str(x).strip().lower()

def parse_conf(parsed: Dict[str, Any]) -> Dict[str, float]:
    def get_conf(section: str) -> float:
        try:
            return float(parsed.get(section, {}).get("conf", 0.0))
        except Exception:
            return 0.0
    return {
        "exist": get_conf("exist"),
        "attr": get_conf("attribute"),
        "spatial": get_conf("spatial"),
        "geom": get_conf("geometry"),
    }

def compute_eval(episode: Dict[str, Any], parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      eval_ dict: correctness + failure flags
      parsed_meta: proxy metrics like conf_avg, has_bbox, format_ok, length_norm, used_abstain
    """
    questions = episode["questions"]
    flags = episode["flags"]

    raw_text = parsed.get("_raw", "") if isinstance(parsed, dict) else ""
    format_ok = int(isinstance(parsed, dict) and ("exist" in parsed) and ("abstain" in parsed))

    # abstain flags
    abst = parsed.get("abstain", {}) if isinstance(parsed, dict) else {}
    abst_exist = bool(abst.get("exist", False))
    abst_attr = bool(abst.get("attribute", False))
    abst_spatial = bool(abst.get("spatial", False))
    abst_geom = bool(abst.get("geometry", False))
    used_abstain = int(any([abst_exist, abst_attr, abst_spatial, abst_geom]))

    # --- Existence ---
    q_exist = questions[0]
    gt_exists = bool(q_exist.gt["exists"])
    pred_exist = safe_lower(parsed.get("exist", {}).get("answer", "")) if isinstance(parsed, dict) else ""
    pred_yes = (pred_exist == "yes")
    pred_no = (pred_exist == "no")

    exist_correct = int((gt_exists and pred_yes and not abst_exist) or ((not gt_exists) and pred_no and not abst_exist))
    exist_hallucination = int((not gt_exists) and pred_yes and not abst_exist)

    # IoU only if gt exists AND pred yes and bbox provided
    iou_val = 0.0
    if gt_exists and pred_yes and (not abst_exist):
        gt_bbox = q_exist.gt["bbox"]
        pred_bbox = parsed.get("exist", {}).get("bbox", None)
        if isinstance(pred_bbox, list) and len(pred_bbox) == 4 and all(isinstance(v, (int, float)) for v in pred_bbox):
            iou_val = float(iou_fn(list(map(int, pred_bbox)), list(map(int, gt_bbox))))
        else:
            iou_val = 0.0

    # --- Attribute ---
    q_attr = questions[1]
    gt_attr = safe_lower(q_attr.gt["color"])
    pred_attr = safe_lower(parsed.get("attribute", {}).get("answer", "")) if isinstance(parsed, dict) else ""
    attr_correct = int((pred_attr == gt_attr) and (not abst_attr))

    # --- Spatial left/right ---
    q_sp = questions[2]
    gt_sp = bool(q_sp.gt["answer"])
    pred_sp = safe_lower(parsed.get("spatial", {}).get("answer", "")) if isinstance(parsed, dict) else ""
    pred_sp_yes = (pred_sp == "yes")
    pred_sp_no = (pred_sp == "no")
    spatial_correct = int(((gt_sp and pred_sp_yes) or ((not gt_sp) and pred_sp_no)) and (not abst_spatial))

    # spatial flip rate: if wrong and it's a left/right yes/no question
    # flipped == chose the opposite boolean (i.e., answered yes when gt is no or vice versa)
    spatial_flip = int((not spatial_correct) and (pred_sp_yes or pred_sp_no) and (not abst_spatial))

    # --- Geometry ---
    q_geom = questions[3]
    gt_geom = safe_lower(q_geom.gt["closer"])
    pred_geom = safe_lower(parsed.get("geometry", {}).get("answer", "")) if isinstance(parsed, dict) else ""
    geom_correct = int((pred_geom == gt_geom) and (not abst_geom))

    # --- Ambiguity-based abstain misuse ---
    # If ambiguous=True, abstain is acceptable; if ambiguous=False, abstain is misuse.
    # Also misuse if ambiguous=True but model does NOT abstain and is wrong (overconfident guessing).
    def abstain_misuse(q_ambiguous: bool, abstained: bool, correct: int) -> int:
        if not q_ambiguous and abstained:
            return 1
        if q_ambiguous and (not abstained) and (correct == 0):
            return 1
        return 0

    misuse = 0
    misuse += abstain_misuse(q_exist.ambiguous, abst_exist, exist_correct)
    misuse += abstain_misuse(q_attr.ambiguous, abst_attr, attr_correct)
    misuse += abstain_misuse(q_sp.ambiguous, abst_spatial, spatial_correct)
    misuse += abstain_misuse(q_geom.ambiguous, abst_geom, geom_correct)

    # relation answered with missing objects (for our synthetic set, objects always exist except q_exist)
    relation_answered_with_missing_objects = 0  # kept for compatibility

    confs = []
    for k in ["exist","attribute","spatial","geometry"]:
        try:
            confs.append(float(parsed.get(k, {}).get("conf", 0.0)))
        except Exception:
            confs.append(0.0)
    conf_avg = sum(confs)/len(confs) if confs else 0.0

    has_bbox = int(isinstance(parsed, dict) and isinstance(parsed.get("exist", {}).get("bbox", None), list))
    length_norm = min(1.0, len(raw_text)/900.0) if raw_text else min(1.0, len(json.dumps(parsed))/900.0)

    # overall accuracy across 4 subquestions (non-abstain correct counts)
    acc_items = [exist_correct, attr_correct, spatial_correct, geom_correct]
    overall_acc = sum(acc_items) / 4.0

    eval_ = {
        "exist_correct": exist_correct,
        "exist_hallucination": exist_hallucination,
        "iou": iou_val,
        "attr_correct": attr_correct,
        "spatial_correct": spatial_correct,
        "spatial_flip": spatial_flip,
        "geom_correct": geom_correct,
        "relation_answered_with_missing_objects": relation_answered_with_missing_objects,
        "abstain_misuse": misuse,
        "overall_acc": overall_acc,
        "conf_avg": conf_avg,
        "ambiguous_any": int(any([q.ambiguous for q in questions])),
    }

    parsed_meta = {
        "format_ok": format_ok,
        "conf_avg": conf_avg,
        "has_bbox": has_bbox,
        "length_norm": length_norm,
        "used_abstain": used_abstain,
    }

    return eval_, parsed_meta