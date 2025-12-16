#!/usr/bin/env python3
"""Measure request-level TTFT, TPOT, E2E latency, and throughput from a vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


async def run_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    row: dict,
    max_tokens: int,
    tokenizer,
) -> dict:
    async with semaphore:
        started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        stream = await client.chat.completions.create(
            model=model,
            messages=row["prompt"],
            temperature=0.0,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(delta)
        finished = time.perf_counter()
        text = "".join(pieces)
        output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        ttft_ms = ((first_token_at or finished) - started) * 1000
        e2e_ms = (finished - started) * 1000
        tpot_ms = None
        if output_tokens > 1:
            tpot_ms = (e2e_ms - ttft_ms) / (output_tokens - 1)
        return {
            "problem_id": row["problem_id"],
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "tpot_ms": tpot_ms,
            "output_tokens": output_tokens,
            "completion": text,
        }


async def async_main(args: argparse.Namespace) -> None:
    rows = [
        json.loads(line)
        for line in args.eval_file.read_text(encoding="utf-8").splitlines()
        if line
    ][: args.num_prompts]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, use_fast=True)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    records = await asyncio.gather(
        *[
            run_request(client, semaphore, args.model, row, args.max_tokens, tokenizer)
            for row in rows
        ]
    )
    wall_time = time.perf_counter() - started
    await client.close()

    ttft = [record["ttft_ms"] for record in records]
    e2e = [record["e2e_ms"] for record in records]
    tpot = [record["tpot_ms"] for record in records if record["tpot_ms"] is not None]
    total_tokens = sum(record["output_tokens"] for record in records)
    summary = {
        "model": args.model,
        "concurrency": args.concurrency,
        "num_prompts": len(records),
        "wall_time_s": wall_time,
        "output_throughput_tokens_s": total_tokens / wall_time if wall_time else 0.0,
        "request_throughput_s": len(records) / wall_time if wall_time else 0.0,
        "mean_output_tokens": statistics.fmean(
            record["output_tokens"] for record in records
        ) if records else 0.0,
        "p50_ttft_ms": percentile(ttft, 0.5),
        "p99_ttft_ms": percentile(ttft, 0.99),
        "p50_tpot_ms": percentile(tpot, 0.5),
        "p99_tpot_ms": percentile(tpot, 0.99),
        "p50_e2e_ms": percentile(e2e, 0.5),
        "p99_e2e_ms": percentile(e2e, 0.99),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", required=True, help="The server's served-model-name")
    parser.add_argument("--tokenizer", help="Tokenizer path if different from served model name")
    parser.add_argument("--eval-file", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

