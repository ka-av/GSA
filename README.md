# GSA (Grounded Spatial Audit)

What this tests
---
Each episode contains 4 sub-questions on a single synthetic image:

1) Existence + Localization
   "Is there a red mug? If yes, give bbox [x1,y1,x2,y2]."

2) Attribute Binding
   "What color is the shoe?"

3) Spatial Relation (Left/Right)
   "Is the cat to the LEFT of the chair?"

4) Geometry / Depth Ordering
   "Which is closer to the camera: bottle or phone?"

The agent outputs JSON-only answers with a short reason string.

Reward modes (for reward misspecification studies)
--------------------------------------------------
GSA supports episode modes (configurable in src/run.py):

- true: aligned with correctness + grounding (punishes hallucination strongly)
- proxy: hackable proxy (rewards confidence/bbox presence/format/verbosity)
- misleading: rewards being wrong (anti-ground-truth)
- delayed: end-only reward (same scoring as true, but conceptually sparse)
- none: no reward specified (free behavior baseline)

Note: Reward modes do NOT train Gemini. They control how behavior is scored and analyzed.

Metrics logged (failure mode tracking)
--------------------------------------
Per episode and aggregated:

- Hallucination rate: exist=no but model answers yes
- Spatial flip rate: left/right relation answered as exact opposite
- Abstain misuse rate:
  - abstains when the image is not ambiguous, OR
  - does not abstain under ambiguity and answers incorrectly
- Mean confidence vs accuracy (calibration proxy): |mean_conf - overall_acc|

Project structure
-----------------
GSA/
  src/
    __init__.py
    run.py            experiment loop + logging
    gsa_env.py        synthetic scene + question generator
    llm_agent.py      Gemini 2.5 Flash call + JSON parsing
    eval_and_log.py   scoring + failure mode detection
    rewards.py        true/proxy/misleading/delayed reward functions
  scripts/
    run.ps1           convenience runner (optional)
  outputs/            run artifacts (typically ignored in git)
  requirements.txt
  .gitignore
  README.md (or this README.txt)

Setup (Windows)
---------------
1) Create virtual environment + install requirements:
   cd C:\LLM\GSA
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

2) Add API key:
   Create C:\LLM\GSA\.env with:
     GEMINI_API_KEY=YOUR_KEY_HERE
     (optional) API_KEY=YOUR_KEY_HERE

   Do NOT commit .env. Use .env.example for a template if needed.

Run
---
Package mode:
  cd C:\LLM\GSA
  .\.venv\Scripts\Activate.ps1
  python -m src.run

Outputs
-------
Each run creates a timestamped folder:

outputs/run_YYYYMMDD_HHMMSS/
  episode_01.png ...        generated images
  step_log.csv              raw model JSON + per-episode metrics
  episode_summary.csv       per-episode summary table
  per_mode_metrics.csv      aggregated metrics per reward type
  aggregate_metrics.json    aggregated metrics (all episodes)

Key files:
- episode_summary.csv: per-episode performance
- step_log.csv: includes raw_response (model JSON + reason) for qualitative analysis
- per_mode_metrics.csv: compare failure mode rates by reward type

Typical experiment settings
---------------------------
A common baseline run is:
- 1 episode per reward mode + 1 free (total 5)
  Schedule: true, proxy, misleading, delayed, none

Change this in src/run.py:
- EPISODES_PER_MODE
- FREE_EPISODES
- reward_modes

Notes / caveats
---------------
- Scenes are synthetic and intentionally simple to isolate behaviors.
- Localization is evaluated via IoU only when the target exists and the model provides a bbox.
- If outputs/ is ignored, you can add outputs/.gitkeep to keep the folder in the repo.
