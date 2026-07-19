#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SNUAI_DATA_ROOT="${SNUAI_DATA_ROOT:-data/snuaichallenge_data}"

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124 \
  torchaudio==2.6.0+cu124 \
  nvidia-cudnn-cu12==9.1.0.70 \
  --extra-index-url https://download.pytorch.org/whl/cu124
python3 -m pip install --no-build-isolation autoawq==0.2.8
python3 -m pip install -r requirements.txt
python3 scripts/check_env.py

python3 -m src.qwen25_zeroshot \
  --mode eval \
  --prediction-method generate \
  --max-rows "${MAX_ROWS:-20}" \
  --max-pixels "${MAX_PIXELS:-200704}" \
  --output-jsonl outputs/qwen25_32b_awq_generate_smoke.jsonl \
  --summary-json outputs/qwen25_32b_awq_generate_smoke_summary.json
