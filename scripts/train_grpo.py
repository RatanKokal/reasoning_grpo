#!/usr/bin/env python3
"""Continue an SFT LoRA adapter with one of three verifier-guided GRPO rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from reasoning_efficiency.rewards import RewardComputer, RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-adapter", type=Path, default=Path("outputs/sft_lora"))
    parser.add_argument("--train-file", type=Path, default=Path("data/grpo_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=["correctness", "fixed", "adaptive"], required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--length-weight", type=float, default=0.25)
    parser.add_argument("--free-tokens", type=int, default=32)
    parser.add_argument("--adaptive-solve-rate", type=float, default=0.5)
    parser.add_argument("--latency-profile", type=Path)
    parser.add_argument("--latency-intercept-ms", type=float, default=0.0)
    parser.add_argument("--latency-per-token-ms", type=float, default=1.0)
    parser.add_argument("--latency-budget-ms", type=float, default=128.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--vllm-server-base-url", default=None)
    return parser.parse_args()


def load_latency_values(args: argparse.Namespace) -> tuple[float, float, float]:
    if args.latency_profile is None:
        return args.latency_intercept_ms, args.latency_per_token_ms, args.latency_budget_ms
    profile = json.loads(args.latency_profile.read_text(encoding="utf-8"))
    return (
        float(profile.get("latency_intercept_ms", args.latency_intercept_ms)),
        float(profile.get("latency_per_token_ms", args.latency_per_token_ms)),
        float(profile.get("latency_budget_ms", args.latency_budget_ms)),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.batch_size % args.num_generations != 0:
        raise ValueError("--batch-size must be divisible by --num-generations")

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_adapter,
        is_trainable=True,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")

    intercept, per_token, budget = load_latency_values(args)
    reward = RewardComputer(
        RewardConfig(
            mode=args.reward_mode,
            length_weight=args.length_weight,
            free_tokens=args.free_tokens,
            max_completion_length=args.max_completion_length,
            adaptive_solve_rate=args.adaptive_solve_rate,
            latency_intercept_ms=intercept,
            latency_per_token_ms=per_token,
            latency_budget_ms=budget,
        )
    )

    config_kwargs = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=not args.fp16,
        fp16=args.fp16,
        tf32=True,
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=4,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        use_vllm=args.use_vllm,
    )
    if args.use_vllm:
        config_kwargs.update(
            vllm_mode=args.vllm_mode,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
        if args.vllm_mode == "server" and args.vllm_server_base_url:
            config_kwargs["vllm_server_base_url"] = args.vllm_server_base_url

    training_args = GRPOConfig(**config_kwargs)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    reward_manifest = {
        "mode": args.reward_mode,
        "length_weight": args.length_weight,
        "free_tokens": args.free_tokens,
        "adaptive_solve_rate": args.adaptive_solve_rate,
        "latency_intercept_ms": intercept,
        "latency_per_token_ms": per_token,
        "latency_budget_ms": budget,
        "seed": args.seed,
    }
    (args.output_dir / "reward_config.json").write_text(
        json.dumps(reward_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.reward_mode} GRPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()

