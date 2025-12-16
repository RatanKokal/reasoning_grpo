"""Reward functions for correctness, fixed-length, and adaptive-latency GRPO."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from .answers import answers_equal, extract_final_answer


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for a composite verifier reward."""

    mode: str = "correctness"
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    format_bonus: float = 0.1
    length_weight: float = 0.25
    free_tokens: int = 32
    max_completion_length: int = 256
    adaptive_solve_rate: float = 0.5
    latency_intercept_ms: float = 0.0
    latency_per_token_ms: float = 1.0
    latency_budget_ms: float = 128.0

    def __post_init__(self) -> None:
        if self.mode not in {"correctness", "fixed", "adaptive"}:
            raise ValueError(f"Unsupported reward mode: {self.mode}")
        if self.max_completion_length <= 0:
            raise ValueError("max_completion_length must be positive")
        if not 0.0 <= self.adaptive_solve_rate <= 1.0:
            raise ValueError("adaptive_solve_rate must be in [0, 1]")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")


class RewardComputer:
    """Picklable callable compatible with TRL GRPOTrainer custom rewards.

    Modes:
      correctness: verifier reward plus a small format bonus.
      fixed: additionally penalize excess length for correct completions.
      adaptive: apply a latency-calibrated penalty only to groups whose empirical
                solve rate is at least adaptive_solve_rate.
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.__name__ = f"{config.mode}_reward"

    def __call__(
        self,
        completions: Sequence[Any],
        ground_truth: Sequence[str],
        completion_ids: Sequence[Sequence[int]] | None = None,
        problem_id: Sequence[str] | None = None,
        log_extra: Any = None,
        log_metric: Any = None,
        **_: Any,
    ) -> list[float]:
        texts = [_completion_text(completion) for completion in completions]
        predictions = [extract_final_answer(text) for text in texts]
        correct = [answers_equal(pred, ref) for pred, ref in zip(predictions, ground_truth)]
        formatted = ["<answer>" in text.lower() and "</answer>" in text.lower() for text in texts]

        if completion_ids is None:
            lengths = [max(1, len(text.split())) for text in texts]
        else:
            lengths = [len(ids) for ids in completion_ids]

        group_solve_rates = self._group_solve_rates(correct, problem_id, ground_truth)
        rewards: list[float] = []
        penalties: list[float] = []

        for index, (is_correct, has_format, length) in enumerate(
            zip(correct, formatted, lengths)
        ):
            reward = self.config.correct_reward if is_correct else self.config.incorrect_reward
            if has_format:
                reward += self.config.format_bonus

            penalty = 0.0
            if is_correct and self.config.mode == "fixed":
                excess = max(0, length - self.config.free_tokens)
                denominator = max(1, self.config.max_completion_length - self.config.free_tokens)
                penalty = self.config.length_weight * excess / denominator
            elif is_correct and self.config.mode == "adaptive":
                group_rate = group_solve_rates[index]
                if group_rate >= self.config.adaptive_solve_rate:
                    predicted_latency = (
                        self.config.latency_intercept_ms
                        + self.config.latency_per_token_ms * length
                    )
                    excess_ratio = max(
                        0.0,
                        (predicted_latency - self.config.latency_budget_ms)
                        / self.config.latency_budget_ms,
                    )
                    penalty = self.config.length_weight * excess_ratio

            rewards.append(float(reward - penalty))
            penalties.append(float(penalty))

        if log_extra is not None:
            log_extra("extracted_answer", [str(value) for value in predictions])
            log_extra("is_correct", [int(value) for value in correct])
            log_extra("completion_tokens", lengths)
            log_extra("efficiency_penalty", penalties)
        if log_metric is not None and rewards:
            log_metric("reward/accuracy", sum(correct) / len(correct))
            log_metric("reward/mean_tokens", sum(lengths) / len(lengths))
            log_metric("reward/mean_efficiency_penalty", sum(penalties) / len(penalties))
        return rewards

    @staticmethod
    def _group_solve_rates(
        correct: Sequence[bool],
        problem_id: Sequence[str] | None,
        ground_truth: Sequence[str],
    ) -> list[float]:
        # TRL repeats non-prompt dataset columns for every sampled completion. A
        # stable problem id is therefore the cleanest way to reconstruct groups.
        keys: Sequence[str] = problem_id if problem_id is not None else ground_truth
        totals: dict[str, int] = defaultdict(int)
        solved: dict[str, int] = defaultdict(int)
        for key, is_correct in zip(keys, correct):
            key = str(key)
            totals[key] += 1
            solved[key] += int(is_correct)
        return [solved[str(key)] / totals[str(key)] for key in keys]

