#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m src.qwen25_zeroshot \
  --mode eval \
  --prediction-method generate \
  --max-rows 20 \
  --max-pixels 200704 \
  --output-jsonl outputs/qwen25_32b_awq_generate_val_20.jsonl \
  --summary-json outputs/qwen25_32b_awq_generate_val_20_summary.json
