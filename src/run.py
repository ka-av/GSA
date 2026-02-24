from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from .gsa_env import make_episode
from .llm_agent import call_gemini
from .eval_and_log import compute_eval
from .rewards import reward_true, reward_proxy, reward_misleading, reward_delayed


def main():
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

    api_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("API_KEY", "").strip()
    )

    if not api_key or "PASTE_YOUR_KEY_HERE" in api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY in .env "
            "or as an environment variable before running."
        )
    
    # 4 episodes per mode (true/proxy/misleading/delayed) + 1 free episode
    reward_modes = ["true", "proxy", "misleading", "delayed"]
    EPISODES_PER_MODE = 1
    FREE_EPISODES = 1

    reward_schedule = []
    for mode in reward_modes:
        reward_schedule += [mode] * EPISODES_PER_MODE
    reward_schedule += ["none"] * FREE_EPISODES  # final episode: no reward mode

    TOTAL_EPISODES = len(reward_schedule)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    step_rows = []
    epi_rows = []

    for ep in tqdm(range(TOTAL_EPISODES), desc="Episodes"):
        reward_type = reward_schedule[ep]
        seed = 1000 + ep  # deterministic
        episode = make_episode(seed)

        # save image
        img_path = out_dir / f"episode_{ep+1:02d}.png"
        episode["image"].save(img_path)

        questions = [q.prompt for q in episode["questions"]]

        raw_text, parsed = call_gemini(
            image=episode["image"],
            questions=questions,
            api_key=api_key,
            model="gemini-2.5-flash"
        )

        eval_, meta = compute_eval(episode, parsed)

        # Compute reward only if reward mode is specified
        R = float("nan")
        if reward_type == "true":
            R = reward_true(eval_)
        elif reward_type == "proxy":
            R = reward_proxy(meta, eval_)
        elif reward_type == "misleading":
            R = reward_misleading(eval_, meta)
        elif reward_type == "delayed":
            # end-only reward equals TRUE score, but conceptually delayed
            end_score = reward_true(eval_)
            R = reward_delayed(end_score)
        elif reward_type == "none":
            # free episode: do NOT compute reward
            pass
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        # Failure mode stats (logged for all episodes, including "none")
        halluc = eval_["exist_hallucination"]
        flip = eval_["spatial_flip"]
        abst_misuse = eval_["abstain_misuse"]
        conf_avg = eval_["conf_avg"]
        acc = eval_["overall_acc"]

        step_rows.append({
            "episode": ep + 1,
            "reward_type": reward_type,
            "seed": seed,
            "image_path": str(img_path),
            "raw_response": raw_text,
            "parsed_ok": int(isinstance(parsed, dict) and not parsed.get("_parse_error", False)),
            "reward": R,
            "exist_correct": eval_["exist_correct"],
            "exist_hallucination": halluc,
            "iou": eval_["iou"],
            "attr_correct": eval_["attr_correct"],
            "spatial_correct": eval_["spatial_correct"],
            "spatial_flip": flip,
            "geom_correct": eval_["geom_correct"],
            "abstain_misuse": abst_misuse,
            "conf_avg": conf_avg,
            "overall_acc": acc,
            "ambiguous_any": eval_["ambiguous_any"],
        })

        epi_rows.append({
            "episode": ep + 1,
            "reward_type": reward_type,
            "reward": R,
            "overall_acc": acc,
            "mean_conf": conf_avg,
            "calibration_gap": abs(conf_avg - acc),
            "hallucination_rate": halluc,            # binary per-episode here
            "spatial_flip_rate": flip,               # binary per-episode here
            "abstain_misuse_rate": abst_misuse / 4.0,  # normalized by 4 subQs
            "iou": eval_["iou"],
        })

    # Write logs
    step_df = pd.DataFrame(step_rows)
    epi_df = pd.DataFrame(epi_rows)

    step_df.to_csv(out_dir / "step_log.csv", index=False)
    epi_df.to_csv(out_dir / "episode_summary.csv", index=False)

    # Aggregate metrics (exclude NaN rewards from mean_reward)
    agg = {
        "episodes": TOTAL_EPISODES,
        "episodes_per_mode": EPISODES_PER_MODE,
        "free_episodes": FREE_EPISODES,
        "hallucination_rate_mean": float(step_df["exist_hallucination"].mean()),
        "spatial_flip_rate_mean": float(step_df["spatial_flip"].mean()),
        "abstain_misuse_rate_mean": float((step_df["abstain_misuse"] / 4.0).mean()),
        "mean_conf": float(step_df["conf_avg"].mean()),
        "mean_acc": float(step_df["overall_acc"].mean()),
        "mean_calibration_gap": float((step_df["conf_avg"] - step_df["overall_acc"]).abs().mean()),
        "mean_reward_excluding_free": float(step_df["reward"].dropna().mean()),
    }

    # Also compute per-reward-mode aggregates (including "none" but reward stats will be NaN)
    per_mode = []
    for mode in ["true", "proxy", "misleading", "delayed", "none"]:
        sdf = step_df[step_df["reward_type"] == mode]
        if len(sdf) == 0:
            continue
        per_mode.append({
            "reward_type": mode,
            "episodes": int(len(sdf)),
            "hallucination_rate": float(sdf["exist_hallucination"].mean()),
            "spatial_flip_rate": float(sdf["spatial_flip"].mean()),
            "abstain_misuse_rate": float((sdf["abstain_misuse"] / 4.0).mean()),
            "mean_conf": float(sdf["conf_avg"].mean()),
            "mean_acc": float(sdf["overall_acc"].mean()),
            "mean_calibration_gap": float((sdf["conf_avg"] - sdf["overall_acc"]).abs().mean()),
            "mean_reward": float(sdf["reward"].dropna().mean()) if sdf["reward"].notna().any() else float("nan"),
        })

    per_mode_df = pd.DataFrame(per_mode)
    per_mode_df.to_csv(out_dir / "per_mode_metrics.csv", index=False)

    # Print summary tables
    print("\n=== Episode Summary ===")
    print(epi_df[[
        "episode", "reward_type", "reward", "overall_acc", "mean_conf", "calibration_gap",
        "hallucination_rate", "spatial_flip_rate", "abstain_misuse_rate", "iou"
    ]].to_string(index=False))

    print("\n=== Per-Mode Metrics ===")
    print(per_mode_df.to_string(index=False))

    print("\n=== Aggregate Metrics ===")
    for k, v in agg.items():
        print(f"{k}: {v}")

    with open(out_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_mode": per_mode}, f, indent=2)

    print(f"\nSaved logs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()