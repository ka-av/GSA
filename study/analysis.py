#!/usr/bin/env python3
"""
Post-hoc analysis of the GSA Comprehension Study.

Produces tables and statistics for the paper:
  1. Exist accuracy by condition
  2. Colour mention adaptation curve (first-half vs second-half)
  3. Colour shift analysis (which colours get preferred over time)
  4. Normal vs Delayed comparison (does immediate feedback help learning?)
  5. Per-episode adaptation trajectory
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
import statistics


def load_results(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ["episode", "block", "seed", "gt_exists", "pred_yes",
                       "exist_correct", "color_mentioned"]:
                if k in r:
                    r[k] = int(r[k]) if r[k] not in ("", "-1") else -1
            for k in ["r_true", "r_proxy", "r_misleading", "r_total"]:
                if k in r:
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def adaptation_curve(rows: List[Dict], condition: str) -> Dict[int, Dict[str, float]]:
    """Per-episode metrics averaged across blocks for a single condition."""
    by_ep = defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            by_ep[r["episode"]].append(r)

    curve = {}
    for ep in sorted(by_ep):
        rs = by_ep[ep]
        n = len(rs)
        curve[ep] = {
            "exist_accuracy": sum(r["exist_correct"] for r in rs) / n,
            "color_mention_rate": sum(r["color_mentioned"] for r in rs) / n,
            "mean_total_reward": sum(r["r_total"] for r in rs) / n,
            "color_dist": dict(Counter(r["stated_color"] for r in rs if r["stated_color"])),
        }
    return curve


def normal_vs_delayed_table(rows: List[Dict]) -> str:
    """Markdown table comparing normal vs delayed across both run configs."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)

    lines = [
        "| Condition | N | Exist Acc | Color Rate | Color Rate (1st½) | Color Rate (2nd½) | Δ Color | Mean R_total | Mean R_true | Mean R_proxy | Mean R_misleading |",
        "|-----------|---|-----------|------------|--------------------|--------------------|---------|--------------|-------------|--------------|-------------------|",
    ]

    for cond in ["normal_run1", "normal_run2", "delayed_run1", "delayed_run2"]:
        rs = groups.get(cond, [])
        n = len(rs)
        if n == 0:
            continue
        half = n // 2
        first_half = rs[:half]
        second_half = rs[half:]
        cr_1 = sum(r["color_mentioned"] for r in first_half) / max(1, len(first_half))
        cr_2 = sum(r["color_mentioned"] for r in second_half) / max(1, len(second_half))
        lines.append(
            f"| {cond:<15s} | {n} "
            f"| {sum(r['exist_correct'] for r in rs)/n:.2f}     "
            f"| {sum(r['color_mentioned'] for r in rs)/n:.2f}      "
            f"| {cr_1:.2f}               "
            f"| {cr_2:.2f}               "
            f"| {cr_2 - cr_1:+.2f}   "
            f"| {sum(r['r_total'] for r in rs)/n:+.1f}         "
            f"| {sum(r['r_true'] for r in rs)/n:+.1f}        "
            f"| {sum(r['r_proxy'] for r in rs)/n:+.1f}        "
            f"| {sum(r['r_misleading'] for r in rs)/n:+.1f}             |"
        )
    return "\n".join(lines)


def colour_shift_analysis(rows: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """For each condition: which colours are mentioned in first vs second half."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)

    analysis = {}
    for cond, rs in groups.items():
        half = len(rs) // 2
        c1 = Counter(r["stated_color"] for r in rs[:half] if r["stated_color"])
        c2 = Counter(r["stated_color"] for r in rs[half:] if r["stated_color"])
        analysis[cond] = {
            "first_half_colors": dict(c1),
            "second_half_colors": dict(c2),
            "shift_direction": "toward_rewarded" if _check_shift(cond, c1, c2) else "no_clear_shift",
        }
    return analysis


def _check_shift(cond: str, c1: Counter, c2: Counter) -> bool:
    """Heuristic: did rewarded colours increase in second half?"""
    rewarded = {"red", "blue"}
    r1 = sum(c1.get(c, 0) for c in rewarded)
    r2 = sum(c2.get(c, 0) for c in rewarded)
    t1 = sum(c1.values()) or 1
    t2 = sum(c2.values()) or 1
    return (r2 / t2) > (r1 / t1)


def main():
    if len(sys.argv) < 2:
        # Find most recent study
        out = Path("outputs")
        studies = sorted(out.glob("study_*/study_results.csv"))
        if not studies:
            print("No study results found. Run the study first.")
            return
        csv_path = str(studies[-1])
    else:
        csv_path = sys.argv[1]

    print(f"Analysing: {csv_path}\n")
    rows = load_results(csv_path)

    print("=" * 80)
    print("1. NORMAL vs DELAYED COMPARISON")
    print("=" * 80)
    print(normal_vs_delayed_table(rows))

    print(f"\n{'=' * 80}")
    print("2. COLOUR SHIFT ANALYSIS")
    print("=" * 80)
    shift = colour_shift_analysis(rows)
    print(json.dumps(shift, indent=2))

    print(f"\n{'=' * 80}")
    print("3. PER-EPISODE ADAPTATION CURVES")
    print("=" * 80)
    conditions = sorted(set(r["condition"] for r in rows))
    for cond in conditions:
        print(f"\n  {cond}:")
        curve = adaptation_curve(rows, cond)
        for ep, metrics in curve.items():
            print(f"    Ep {ep:>2d}: acc={metrics['exist_accuracy']:.2f}  "
                  f"color_rate={metrics['color_mention_rate']:.2f}  "
                  f"R={metrics['mean_total_reward']:+.1f}  "
                  f"colors={metrics['color_dist']}")

    # Save analysis
    analysis_path = Path(csv_path).parent / "analysis.json"
    analysis = {
        "colour_shift": shift,
        "adaptation_curves": {
            cond: adaptation_curve(rows, cond) for cond in conditions
        },
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
