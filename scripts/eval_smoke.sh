#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m src.qwen25_zeroshot \
  --mode eval \
  --load-in-4bit \
  --max-rows 20 \
  --output-jsonl outputs/qwen25_val_20.jsonl \
  --summary-json outputs/qwen25_val_20_summary.json
