#!/usr/bin/env python3
"""Create a compact Markdown comparison from evaluation summary files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/comparison.md"))
    args = parser.parse_args()

    rows = []
    for path in args.summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            (
                path.parent.name,
                data["accuracy"],
                data["mean_output_tokens"],
                data.get("mean_correct_output_tokens"),
                data.get("aggregate_output_tokens_per_s"),
            )
        )
    lines = [
        "# Evaluation comparison",
        "",
        "| Experiment | Accuracy | Mean output tokens | Mean tokens when correct | Aggregate tok/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, accuracy, mean_tokens, correct_tokens, throughput in rows:
        correct_display = "—" if correct_tokens is None else f"{correct_tokens:.1f}"
        throughput_display = "—" if throughput is None else f"{throughput:.1f}"
        lines.append(
            f"| {name} | {accuracy:.3%} | {mean_tokens:.1f} | "
            f"{correct_display} | {throughput_display} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

