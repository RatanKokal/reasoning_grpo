#!/usr/bin/env python3
"""Fit the simple E2E latency model consumed by adaptive GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/latency_profile.json"))
    parser.add_argument(
        "--budget-percentile",
        type=float,
        default=50.0,
        help="Observed E2E percentile used as the adaptive reward budget",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.requests.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) < 10:
        raise ValueError("At least 10 benchmark requests are required for calibration")
    tokens = np.asarray([row["output_tokens"] for row in rows], dtype=np.float64)
    latency = np.asarray([row["e2e_ms"] for row in rows], dtype=np.float64)
    slope, intercept = np.polyfit(tokens, latency, 1)
    predicted = slope * tokens + intercept
    residual = latency - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((latency - latency.mean()) ** 2))
    profile = {
        "latency_intercept_ms": max(0.0, float(intercept)),
        "latency_per_token_ms": max(1e-6, float(slope)),
        "latency_budget_ms": float(np.percentile(latency, args.budget_percentile)),
        "budget_percentile": args.budget_percentile,
        "num_samples": len(rows),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
        "note": "Fit on request-level E2E latency; calibrate at the target concurrency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()

