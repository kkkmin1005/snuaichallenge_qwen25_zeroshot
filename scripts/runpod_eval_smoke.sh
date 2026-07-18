#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SNUAI_DATA_ROOT="${SNUAI_DATA_ROOT:-/workspace/snuaichallenge_data}"

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 -m src.qwen25_zeroshot \
  --mode eval \
  --load-in-4bit \
  --max-rows "${MAX_ROWS:-20}" \
  --output-jsonl outputs/qwen25_val_smoke.jsonl \
  --summary-json outputs/qwen25_val_smoke_summary.json
