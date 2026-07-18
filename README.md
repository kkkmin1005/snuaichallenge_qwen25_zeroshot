# SNU AI Challenge Qwen2.5-VL Zero-Shot

This is a separate zero-shot runner for `Qwen/Qwen2.5-VL-7B-Instruct`.
It reads the existing dataset from:

```text
/home/kangmin/snuaichallenge/snuaichallenge_data
```

## Install

```bash
cd /home/kangmin/snuaichallenge_qwen25_zeroshot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/check_env.py
```

The pinned GPU stack is aligned to the challenge server's CUDA 12.4 target:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
transformers==4.49.0
accelerate==1.4.0
bitsandbytes==0.46.0
qwen-vl-utils[decord]==0.0.8
```

## Quick Validation Smoke Test

Start with a small run because VLM inference is slow.

```bash
python3 -m src.qwen25_zeroshot \
  --mode eval \
  --load-in-4bit \
  --max-rows 20 \
  --output-jsonl outputs/qwen25_val_20.jsonl \
  --summary-json outputs/qwen25_val_20_summary.json
```

## Larger Validation Run

```bash
python3 -m src.qwen25_zeroshot \
  --mode eval \
  --load-in-4bit \
  --val-ratio 0.2 \
  --output-jsonl outputs/qwen25_val.jsonl \
  --summary-json outputs/qwen25_val_summary.json
```

## Test Submission

```bash
python3 -m src.qwen25_zeroshot \
  --mode infer \
  --load-in-4bit \
  --output-csv outputs/qwen25_submission.csv \
  --output-jsonl outputs/qwen25_test.jsonl
```

The submission CSV uses the competition format: `Id,Answer`, where `Answer` is
a one-based permutation such as `[3, 1, 2, 4]`.

## RunPod

Clone this repo on the pod, then put the competition data at:

```text
/workspace/snuaichallenge_data/train.csv
/workspace/snuaichallenge_data/test.csv
/workspace/snuaichallenge_data/train/{Id}/*.jpg
/workspace/snuaichallenge_data/test/{Id}/*.jpg
```

Then run:

```bash
bash scripts/runpod_eval_smoke.sh
```

If the data is somewhere else, set `SNUAI_DATA_ROOT`:

```bash
SNUAI_DATA_ROOT=/path/to/snuaichallenge_data bash scripts/runpod_eval_smoke.sh
```
