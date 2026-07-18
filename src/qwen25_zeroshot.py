from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


PERMUTATIONS: list[list[int]] = [list(perm) for perm in permutations([1, 2, 3, 4])]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("SNUAI_DATA_ROOT", "/home/kangmin/snuaichallenge/snuaichallenge_data")
)


@dataclass(frozen=True)
class Prediction:
    answer: list[int]
    raw_text: str
    parse_error: str | None


def main() -> None:
    args = parse_args()
    processor, model = load_model(args)

    if args.mode == "eval":
        run_eval(args, processor, model)
    elif args.mode == "infer":
        run_infer(args, processor, model)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def load_model(args: argparse.Namespace):
    processor_kwargs: dict[str, Any] = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "torch_dtype": dtype_from_name(args.dtype),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    return processor, model


def run_eval(args: argparse.Namespace, processor, model) -> None:
    df = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    indices = validation_indices(len(df), args.val_ratio, args.seed)
    if args.max_rows is not None:
        indices = indices[: args.max_rows]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "correct": 0,
        "parse_errors": 0,
        "no_ordering_total": 0,
        "no_ordering_correct": 0,
        "ordered_total": 0,
        "ordered_correct": 0,
    }

    with output_path.open("w", encoding="utf-8") as out:
        for index in tqdm(indices, desc="eval"):
            row = df.iloc[index]
            true_answer = parse_order(row[args.answer_col])
            pred = predict_row(args, processor, model, row, split="train")
            correct = pred.answer == true_answer
            no_ordering = bool_value(row.get(args.no_ordering_col, False))

            update_stats(stats, correct, no_ordering, pred.parse_error)
            record = {
                "index": int(index),
                "id": str(row[args.id_col]),
                "sentence": str(row[args.sentence_col]),
                "true_answer": true_answer,
                "pred_answer": pred.answer,
                "correct": correct,
                "no_ordering": no_ordering,
                "raw_text": pred.raw_text,
                "parse_error": pred.parse_error,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(args, stats)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_infer(args: argparse.Namespace, processor, model) -> None:
    df = pd.read_csv(args.test_csv, encoding="utf-8-sig")
    indices = list(range(len(df)))
    if args.max_rows is not None:
        indices = indices[: args.max_rows]

    jsonl_path = Path(args.output_jsonl)
    csv_path = Path(args.output_csv)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with jsonl_path.open("w", encoding="utf-8") as out:
        for index in tqdm(indices, desc="infer"):
            row = df.iloc[index]
            pred = predict_row(args, processor, model, row, split="test")
            row_id = str(row[args.id_col])
            rows.append({args.id_col: row_id, args.answer_col: format_answer(pred.answer)})
            out.write(
                json.dumps(
                    {
                        "index": int(index),
                        "id": row_id,
                        "sentence": str(row[args.sentence_col]),
                        "pred_answer": pred.answer,
                        "raw_text": pred.raw_text,
                        "parse_error": pred.parse_error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[args.id_col, args.answer_col])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path} rows={len(rows)}")


def predict_row(args: argparse.Namespace, processor, model, row: pd.Series, split: str) -> Prediction:
    image_paths = [
        resolve_image_path(
            data_root=args.data_root,
            csv_path=args.train_csv if split == "train" else args.test_csv,
            split=split,
            row_id=str(row[args.id_col]),
            image_name=str(row[col]),
        )
        for col in args.frame_cols
    ]
    messages = build_messages(image_paths, str(row[args.sentence_col]))
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(input_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    answer, error = parse_prediction(raw_text)
    return Prediction(answer=answer, raw_text=raw_text, parse_error=error)


def build_messages(image_paths: list[Path], sentence: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for idx, path in enumerate(image_paths, start=1):
        content.append({"type": "image", "image": str(path)})
        content.append({"type": "text", "text": f"Image {idx}: Input_{idx}"})

    candidates = "\n".join(format_answer(perm) for perm in PERMUTATIONS)
    instruction = f"""
You are solving a four-frame temporal ordering task.
The four images are shuffled video frames named Input_1, Input_2, Input_3, and Input_4.
The sentence describes the correct chronological story:
{sentence}

Choose the chronological order from earliest to latest.
The answer must be exactly one of these 24 one-based permutations:
{candidates}

Return only JSON in this exact format:
{{"answer":[1,2,3,4]}}
""".strip()
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def parse_prediction(text: str) -> tuple[list[int], str | None]:
    try:
        obj = parse_jsonish(text)
        if isinstance(obj, dict) and "answer" in obj:
            answer = normalize_answer(obj["answer"])
        else:
            answer = normalize_answer(obj)
        return answer, None
    except Exception as exc:
        fallback = first_permutation(text)
        if fallback is not None:
            return fallback, f"json_parse_failed: {exc}"
        return [1, 2, 3, 4], f"parse_failed_defaulted_identity: {exc}"


def parse_jsonish(text: str) -> Any:
    brace = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    bracket = re.search(r"\[[^\]]+\]", text)
    if bracket:
        return ast.literal_eval(bracket.group(0))
    return ast.literal_eval(text)


def first_permutation(text: str) -> list[int] | None:
    nums = [int(match) for match in re.findall(r"\b[1-4]\b", text)]
    for start in range(0, max(0, len(nums) - 3)):
        candidate = nums[start : start + 4]
        if sorted(candidate) == [1, 2, 3, 4]:
            return candidate
    return None


def normalize_answer(value: Any) -> list[int]:
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    answer = [int(x) for x in parsed]
    if sorted(answer) == [0, 1, 2, 3]:
        answer = [x + 1 for x in answer]
    if len(answer) != 4 or sorted(answer) != [1, 2, 3, 4]:
        raise ValueError(f"Not a permutation of 1..4: {value!r}")
    return answer


def parse_order(value: Any) -> list[int]:
    return normalize_answer(value)


def resolve_image_path(data_root: Path, csv_path: Path, split: str, row_id: str, image_name: str) -> Path:
    candidates = [
        data_root / split / row_id / image_name,
        data_root / row_id / image_name,
        data_root / image_name,
        csv_path.parent / split / row_id / image_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image {image_name!r} for row {row_id!r}")


def validation_indices(n_rows: int, val_ratio: float, seed: int) -> list[int]:
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(n_rows * val_ratio)))
    return indices[:n_val]


def update_stats(stats: dict[str, int], correct: bool, no_ordering: bool, parse_error: str | None) -> None:
    stats["total"] += 1
    stats["correct"] += int(correct)
    stats["parse_errors"] += int(parse_error is not None)
    if no_ordering:
        stats["no_ordering_total"] += 1
        stats["no_ordering_correct"] += int(correct)
    else:
        stats["ordered_total"] += 1
        stats["ordered_correct"] += int(correct)


def summarize(args: argparse.Namespace, stats: dict[str, int]) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "train_csv": str(args.train_csv),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "total": stats["total"],
        "correct": stats["correct"],
        "accuracy": safe_div(stats["correct"], stats["total"]),
        "parse_errors": stats["parse_errors"],
        "parse_error_rate": safe_div(stats["parse_errors"], stats["total"]),
        "no_ordering_total": stats["no_ordering_total"],
        "no_ordering_accuracy": safe_div(stats["no_ordering_correct"], stats["no_ordering_total"]),
        "ordered_total": stats["ordered_total"],
        "ordered_accuracy": safe_div(stats["ordered_correct"], stats["ordered_total"]),
    }


def safe_div(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def format_answer(answer: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in answer) + "]"


def input_device(model) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return next(model.parameters()).device


def dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "infer"], required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_DATA_ROOT / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_DATA_ROOT / "test.csv")
    parser.add_argument("--output-jsonl", default="outputs/qwen25_outputs.jsonl")
    parser.add_argument("--summary-json", default="outputs/qwen25_summary.json")
    parser.add_argument("--output-csv", default="outputs/qwen25_submission.csv")
    parser.add_argument("--id-col", default="Id")
    parser.add_argument("--frame-cols", nargs=4, default=["Input_1", "Input_2", "Input_3", "Input_4"])
    parser.add_argument("--sentence-col", default="Sentence")
    parser.add_argument("--answer-col", default="Answer")
    parser.add_argument("--no-ordering-col", default="No_ordering")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
