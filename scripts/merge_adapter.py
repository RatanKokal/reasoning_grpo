#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model for vLLM evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    merged = model.merge_and_unload(safe_merge=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()

