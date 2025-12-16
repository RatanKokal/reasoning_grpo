#!/usr/bin/env bash
set -euo pipefail

python scripts/train_grpo.py \
  --reward-mode correctness \
  --output-dir outputs/grpo_correctness \
  --max-steps 225

python scripts/train_grpo.py \
  --reward-mode fixed \
  --output-dir outputs/grpo_fixed \
  --max-steps 225

adaptive_args=()
if [[ -f results/latency_profile.json ]]; then
  adaptive_args+=(--latency-profile results/latency_profile.json)
fi
python scripts/train_grpo.py \
  --reward-mode adaptive \
  --output-dir outputs/grpo_adaptive \
  --max-steps 225 \
  "${adaptive_args[@]}"

