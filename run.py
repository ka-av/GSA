from __future__ import annotations

import os
import json
from datetime import datetime
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from gsa_env import make_episode
from llm_agent import call_agent, reflect_plan
from eval_and_log import compute_eval
from rewards import compute_rewards


def main():
    # Load .env from current directory if present
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    api_key = (os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("API_KEY", "").strip() or None)

    N_NORMAL = 4
    N_DELAYED = 1
    TOTAL = N_NORMAL + N_DELAYED

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    agent_plan = ""
    base_seed = int(os.getenv("RUN_SEED", "0") or "0")
    if base_seed == 0:
        base_seed = random.SystemRandom().randint(1, 10**9-1)

    for ep in tqdm(range(TOTAL), desc="Episodes"):
        seed = base_seed + ep
        episode = make_episode(seed)

        img_path = out_dir / f"episode_{ep+1:02d}.png"
        episode["image"].save(img_path)

        # Episodes 1-4: immediate reward setting (agent knows reward rules)
        # Episode 5: delayed reward setting (agent does NOT see reward rules before answering)
        delayed = (ep == TOTAL - 1)
        include_reward_instructions = (not delayed)

        raw_text, parsed = call_agent(
            image=episode["image"],
            question=episode["question"],
            include_reward_instructions=include_reward_instructions,
            api_key=api_key,
            model="gemini-2.5-flash",
            temperature=0.9,
            rng_seed=seed,
            fallback_context={"gt_exists": episode["gt_exists"], "target": episode["target"]},
            agent_plan=agent_plan,
        )

        eval_, meta = compute_eval(episode, raw_text, parsed if isinstance(parsed, dict) else {})

        # rewards
        rewards = compute_rewards(
            exist_correct=bool(eval_["exist_correct"]),
            color_mentioned=bool(eval_["color_mentioned"]),
            stated_color=(eval_["stated_color"] or None),
        )

        # immediate feedback / plan update for first 4 episodes only
        plan_raw = ""
        plan_obj = {}
        if not delayed:
            feedback = {
                "episode": ep + 1,
                "question": episode["question"],
                "gt_exists": bool(episode["gt_exists"]),
                "model_answer": eval_["pred_answer"],
                "stated_color": eval_["stated_color"],
                "reward_breakdown": rewards,
                "fail_exist_wrong": int(eval_["fail_exist_wrong"]),
                "fail_color_wrong": int(eval_["fail_color_wrong"]),
            }
            plan_raw, plan_obj = reflect_plan(
                last_feedback=feedback,
                api_key=api_key,
                model="gemini-2.5-flash",
                temperature=0.6,
            )
            if isinstance(plan_obj, dict):
                agent_plan = str(plan_obj.get("plan", agent_plan))

        rows.append({
            "episode": ep + 1,
            "seed": seed,
            "delayed_reward_episode": int(delayed),
            "image_path": str(img_path),
            "question": episode["question"],
            "target": episode["target"],
            "present": ",".join(episode["present"]),
            "gt_exists": int(episode["gt_exists"]),
            "target_color": episode["target_color"],
            "raw_response": raw_text,
            "parsed_ok": int(eval_["parsed_ok"]),
            "backend": meta.get("backend", "unknown"),
            "pred_answer": eval_["pred_answer"],
            "exist_correct": int(eval_["exist_correct"]),
            "color_mentioned": int(eval_["color_mentioned"]),
            "stated_color": eval_["stated_color"],
            "color_reason": meta.get("color_reason", ""),
            "exist_reason": meta.get("exist_reason", ""),
            "r_true": rewards["true"],
            "r_proxy": rewards["proxy"],
            "r_misleading": rewards["misleading"],
            "r_total": rewards["total"],
            "fail_exist_wrong": int(eval_["fail_exist_wrong"]),
            "fail_color_wrong": int(eval_["fail_color_wrong"]),
            "fail_color_spam": int(eval_["fail_color_spam"]),
            "plan_after_episode": (str(plan_obj.get("plan", "")) if isinstance(plan_obj, dict) else ""),
            "plan_raw": plan_raw,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "step_log.csv", index=False)

    # aggregate summary
    summary = {
        "episodes": TOTAL,
        "normal_episodes": N_NORMAL,
        "delayed_episodes": N_DELAYED,
        "backend_counts": df["backend"].value_counts().to_dict(),
        "exist_accuracy": float(df["exist_correct"].mean()),
        "color_mention_rate": float(df["color_mentioned"].mean()),
        "ungrounded_color_rate_when_exists": float(
            df.loc[df["gt_exists"] == 1, "fail_color_wrong"].mean() if (df["gt_exists"] == 1).any() else 0.0
        ),
        "mean_total_reward": float(df["r_total"].mean()),
        "mean_true_reward": float(df["r_true"].mean()),
        "mean_proxy_reward": float(df["r_proxy"].mean()),
        "mean_misleading_reward": float(df["r_misleading"].mean()),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved logs to:", out_dir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
