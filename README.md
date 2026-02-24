@"
# GSA (Grounded Spatial Audit) — VLM Failure Modes

Runs a small synthetic VLM benchmark using Gemini 2.5 Flash with reward modes:
- true
- proxy
- misleading
- delayed
- none (free episode)

## Setup
1) Create `.env` in `GSA/`:
   GEMINI_API_KEY=...

2) Create venv + install:
   pip install -r requirements.txt

## Run
python -m src.run

## Outputs
Writes logs under `outputs/run_*/`:
- step_log.csv
- episode_summary.csv
- per_mode_metrics.csv
- aggregate_metrics.json
"@ | Set-Content -Encoding UTF8 C:\LLM\LLM\GSA\README.md