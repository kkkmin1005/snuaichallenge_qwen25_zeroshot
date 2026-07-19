# SNU AI Challenge Qwen2.5-VL Zero-Shot

This is a separate zero-shot runner for `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`.
The default prediction path uses direct generation, not 24-candidate likelihood
scoring.

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
autoawq==0.2.9
qwen-vl-utils[decord]==0.0.8
```

`Qwen/Qwen2.5-VL-32B-Instruct-AWQ` is already a 4-bit AWQ checkpoint. Do not
pass `--load-in-4bit` with this model. Use `--load-in-4bit` only when falling
back to non-AWQ checkpoints such as `Qwen/Qwen2.5-VL-7B-Instruct`.

## Quick Validation Smoke Test

Start with a small run because VLM inference is slow.

```bash
python3 -m src.qwen25_zeroshot \
  --mode eval \
  --prediction-method generate \
  --max-rows 20 \
  --max-pixels 200704 \
  --output-jsonl outputs/qwen25_32b_awq_generate_val_20.jsonl \
  --summary-json outputs/qwen25_32b_awq_generate_val_20_summary.json
```

## Larger Validation Run

```bash
python3 -m src.qwen25_zeroshot \
  --mode eval \
  --prediction-method generate \
  --val-ratio 0.2 \
  --max-pixels 200704 \
  --output-jsonl outputs/qwen25_32b_awq_generate_val.jsonl \
  --summary-json outputs/qwen25_32b_awq_generate_val_summary.json
```

## Test Submission

```bash
python3 -m src.qwen25_zeroshot \
  --mode infer \
  --prediction-method generate \
  --max-pixels 200704 \
  --output-csv outputs/qwen25_32b_awq_submission.csv \
  --output-jsonl outputs/qwen25_32b_awq_test.jsonl
```

The submission CSV uses the competition format: `Id,Answer`, where `Answer` is
a one-based permutation such as `[3, 1, 2, 4]`.

## RunPod

Clone this repo on the pod, then put the competition data at:

```text
data/snuaichallenge_data/train.csv
data/snuaichallenge_data/test.csv
data/snuaichallenge_data/train/{Id}/*.jpg
data/snuaichallenge_data/test/{Id}/*.jpg
```

Then run:

```bash
bash scripts/runpod_eval_smoke.sh
```

If the data is somewhere else, set `SNUAI_DATA_ROOT`:

```bash
SNUAI_DATA_ROOT=/path/to/snuaichallenge_data bash scripts/runpod_eval_smoke.sh
```

## 7B Fallback

If 32B-AWQ does not fit on the GPU with four images, fall back to the 7B model:

```bash
python3 -m src.qwen25_zeroshot \
  --mode eval \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --load-in-4bit \
  --prediction-method generate \
  --max-rows 20 \
  --max-pixels 200704 \
  --output-jsonl outputs/qwen25_7b_4bit_generate_val_20.jsonl \
  --summary-json outputs/qwen25_7b_4bit_generate_val_20_summary.json
```

## Candidate Scoring

The old 24-candidate scoring path is still available for comparison, but it is
much slower because it runs likelihood evaluation over every permutation:

```bash
python3 -m src.qwen25_zeroshot \
  --mode eval \
  --prediction-method score \
  --candidate-batch-size 1 \
  --max-rows 20 \
  --max-pixels 200704 \
  --output-jsonl outputs/qwen25_score_val_20.jsonl \
  --summary-json outputs/qwen25_score_val_20_summary.json
```
