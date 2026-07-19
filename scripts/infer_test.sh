#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m src.qwen25_zeroshot \
  --mode infer \
  --prediction-method generate \
  --max-pixels 200704 \
  --output-csv outputs/qwen25_32b_awq_submission.csv \
  --output-jsonl outputs/qwen25_32b_awq_test.jsonl
