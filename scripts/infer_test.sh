#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m src.qwen25_zeroshot \
  --mode infer \
  --load-in-4bit \
  --output-csv outputs/qwen25_submission.csv \
  --output-jsonl outputs/qwen25_test.jsonl
