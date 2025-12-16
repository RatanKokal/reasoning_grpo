#!/usr/bin/env python3
"""Create disjoint SFT, GRPO, and evaluation splits from GSM8K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from reasoning_efficiency.answers import extract_gsm8k_reference

SYSTEM_PROMPT = (
    "Solve the arithmetic word problem. Give a concise, logically complete derivation. "
    "Return exactly <reasoning>...</reasoning> followed by <answer>...</answer>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sft-size", type=int, default=1500)
    parser.add_argument("--grpo-size", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def main() -> None:
    args = parse_args()
    raw = load_dataset("openai/gsm8k", "main")
    shuffled = raw["train"].shuffle(seed=args.seed)
    required = args.sft_size + args.grpo_size
    if required > len(shuffled):
        raise ValueError(f"Requested {required} train examples, but GSM8K has {len(shuffled)}")
    if args.eval_size > len(raw["test"]):
        raise ValueError(f"Requested {args.eval_size} eval examples, but test has {len(raw['test'])}")

    sft_rows: list[dict] = []
    for index, example in enumerate(shuffled.select(range(args.sft_size))):
        reasoning, final = extract_gsm8k_reference(example["answer"])
        sft_rows.append(
            {
                "problem_id": f"sft-{index}",
                "prompt": prompt(example["question"]),
                "completion": [
                    {
                        "role": "assistant",
                        "content": f"<reasoning>{reasoning}</reasoning>\n<answer>{final}</answer>",
                    }
                ],
            }
        )

    grpo_rows: list[dict] = []
    start = args.sft_size
    for index, example in enumerate(shuffled.select(range(start, start + args.grpo_size))):
        _, final = extract_gsm8k_reference(example["answer"])
        grpo_rows.append(
            {
                "problem_id": f"grpo-{index}",
                "prompt": prompt(example["question"]),
                "ground_truth": final,
                "question": example["question"].strip(),
            }
        )

    eval_rows: list[dict] = []
    for index, example in enumerate(raw["test"].select(range(args.eval_size))):
        _, final = extract_gsm8k_reference(example["answer"])
        eval_rows.append(
            {
                "problem_id": f"eval-{index}",
                "prompt": prompt(example["question"]),
                "ground_truth": final,
                "question": example["question"].strip(),
            }
        )

    write_jsonl(args.output_dir / "sft_train.jsonl", sft_rows)
    write_jsonl(args.output_dir / "grpo_train.jsonl", grpo_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    manifest = {
        "dataset": "openai/gsm8k/main",
        "seed": args.seed,
        "sft_size": len(sft_rows),
        "grpo_size": len(grpo_rows),
        "eval_size": len(eval_rows),
        "disjoint_train_splits": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

