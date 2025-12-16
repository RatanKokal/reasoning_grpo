import pytest

from reasoning_efficiency.rewards import RewardComputer, RewardConfig


def make_completions():
    return [
        "<reasoning>short</reasoning><answer>4</answer>",
        "<reasoning>" + "long " * 60 + "</reasoning><answer>4</answer>",
        "<reasoning>wrong</reasoning><answer>5</answer>",
        "<reasoning>wrong</reasoning><answer>6</answer>",
    ]


def test_correctness_reward():
    reward = RewardComputer(RewardConfig(mode="correctness"))
    values = reward(
        completions=make_completions(),
        ground_truth=["4"] * 4,
        completion_ids=[[1] * 10, [1] * 70, [1] * 10, [1] * 10],
        problem_id=["p1"] * 4,
    )
    assert values[0] == pytest.approx(1.1)
    assert values[1] == pytest.approx(1.1)
    assert values[2] == pytest.approx(-0.9)


def test_fixed_penalty_prefers_short_correct_completion():
    reward = RewardComputer(
        RewardConfig(mode="fixed", length_weight=0.5, free_tokens=16, max_completion_length=80)
    )
    values = reward(
        completions=make_completions(),
        ground_truth=["4"] * 4,
        completion_ids=[[1] * 10, [1] * 70, [1] * 10, [1] * 10],
        problem_id=["p1"] * 4,
    )
    assert values[0] > values[1]


def test_adaptive_penalty_activates_at_solve_rate_threshold():
    reward = RewardComputer(
        RewardConfig(
            mode="adaptive",
            length_weight=0.5,
            adaptive_solve_rate=0.5,
            latency_per_token_ms=2.0,
            latency_budget_ms=40.0,
        )
    )
    values = reward(
        completions=make_completions(),
        ground_truth=["4"] * 4,
        completion_ids=[[1] * 10, [1] * 70, [1] * 10, [1] * 10],
        problem_id=["p1"] * 4,
    )
    assert values[0] == pytest.approx(1.1)
    assert values[1] < values[0]


def test_adaptive_penalty_stays_off_for_hard_group():
    reward = RewardComputer(
        RewardConfig(
            mode="adaptive",
            adaptive_solve_rate=0.75,
            latency_per_token_ms=2.0,
            latency_budget_ms=40.0,
        )
    )
    values = reward(
        completions=make_completions(),
        ground_truth=["4"] * 4,
        completion_ids=[[1] * 10, [1] * 70, [1] * 10, [1] * 10],
        problem_id=["p1"] * 4,
    )
    assert values[0] == pytest.approx(values[1])

