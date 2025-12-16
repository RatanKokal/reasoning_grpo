#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "" ]]; then
  echo "Unset CUDA_VISIBLE_DEVICES before this launcher; it assigns physical GPUs 0, 1, and 2." >&2
  exit 2
fi

adaptive_args=()
if [[ -f results/latency_profile.json ]]; then
  adaptive_args+=(--latency-profile results/latency_profile.json)
fi

CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
  --reward-mode correctness \
  --output-dir outputs/grpo_correctness &
pid_correctness=$!

CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py \
  --reward-mode fixed \
  --output-dir outputs/grpo_fixed &
pid_fixed=$!

CUDA_VISIBLE_DEVICES=2 python scripts/train_grpo.py \
  --reward-mode adaptive \
  --output-dir outputs/grpo_adaptive \
  "${adaptive_args[@]}" &
pid_adaptive=$!

status=0
wait "$pid_correctness" || status=1
wait "$pid_fixed" || status=1
wait "$pid_adaptive" || status=1
exit "$status"

