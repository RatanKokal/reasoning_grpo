#!/usr/bin/env python3
"""Evaluate a base or merged model with vLLM offline inference."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from reasoning_efficiency.answers import answers_equal, extract_final_answer


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows if limit is None else rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.eval_file, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompts = [
        tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    engine = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling, use_tqdm=True)
    elapsed = time.perf_counter() - started

    records: list[dict] = []
    for row, request_output in zip(rows, outputs):
        generated = request_output.outputs[0]
        text = generated.text
        prediction = extract_final_answer(text)
        is_correct = answers_equal(prediction, row["ground_truth"])
        records.append(
            {
                "problem_id": row["problem_id"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "prediction": prediction,
                "correct": is_correct,
                "output_tokens": len(generated.token_ids),
                "completion": text,
            }
        )

    output_tokens = [record["output_tokens"] for record in records]
    correct_tokens = [record["output_tokens"] for record in records if record["correct"]]
    summary = {
        "model": args.model,
        "num_examples": len(records),
        "accuracy": sum(record["correct"] for record in records) / max(1, len(records)),
        "mean_output_tokens": statistics.fmean(output_tokens) if output_tokens else 0.0,
        "median_output_tokens": statistics.median(output_tokens) if output_tokens else 0.0,
        "mean_correct_output_tokens": statistics.fmean(correct_tokens) if correct_tokens else None,
        "wall_time_s": elapsed,
        "aggregate_output_tokens_per_s": sum(output_tokens) / elapsed if elapsed else 0.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

