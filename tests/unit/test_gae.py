from dataclasses import replace

import pytest
import torch

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.gae import compute_gae
from kovara9.training.rollout import RolloutBatch


def _batch(  # noqa: PLR0913
    *,
    rewards: list[float],
    values: list[float],
    next_values: list[float],
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    episode_starts: list[bool] | None = None,
    active_agents: torch.Tensor | None = None,
) -> RolloutBatch:
    time = len(rewards)
    agents = 2 if active_agents is None else active_agents.shape[-1]
    active = (
        torch.ones((time, 1, agents), dtype=torch.bool) if active_agents is None else active_agents
    )
    return RolloutBatch(
        actor_features=torch.zeros((time, 1, agents, 2)),
        critic_features=torch.zeros((time, 1, 3)),
        move_action_masks=torch.ones((time, 1, agents, 2), dtype=torch.bool),
        message_action_masks=torch.ones((time, 1, agents, 2), dtype=torch.bool),
        move_actions=torch.zeros((time, 1, agents), dtype=torch.int64),
        message_actions=torch.zeros((time, 1, agents), dtype=torch.int64),
        move_log_probabilities=torch.zeros((time, 1, agents)),
        message_log_probabilities=torch.zeros((time, 1, agents)),
        joint_log_probabilities=torch.zeros((time, 1, agents)),
        rewards=torch.tensor(rewards).reshape(time, 1),
        values=torch.tensor(values).reshape(time, 1),
        next_values=torch.tensor(next_values).reshape(time, 1),
        terminated=torch.tensor(terminated or [False] * time).reshape(time, 1),
        truncated=torch.tensor(truncated or [False] * time).reshape(time, 1),
        episode_starts=torch.tensor(episode_starts or [False] * time).reshape(time, 1),
        active_agents=active,
        communication_rejections=torch.zeros((time, 1, agents), dtype=torch.bool),
        environment_ids=torch.zeros((time, 1), dtype=torch.int64),
        transition_ids=torch.arange(time, dtype=torch.int64).reshape(time, 1),
        agent_order=tuple(f"agent_{index}" for index in range(agents)),
    )


def _compute(batch: RolloutBatch, *, normalize: bool = False):
    return compute_gae(
        batch,
        gamma=0.9,
        gae_lambda=0.8,
        normalize_advantages=normalize,
        normalization_epsilon=1e-8,
    )


def test_manually_verified_non_terminal_gae() -> None:
    result = _compute(_batch(rewards=[1.0, 1.0], values=[0.0, 0.0], next_values=[0.0, 0.0]))
    expected = torch.tensor([1.72, 1.0]).reshape(2, 1, 1).expand(2, 1, 2)
    assert torch.allclose(result.advantages, expected)
    assert torch.allclose(result.value_targets, expected)


def test_termination_prevents_bootstrap() -> None:
    batch = _batch(
        rewards=[1.0],
        values=[0.5],
        next_values=[10.0],
        terminated=[True],
    )
    result = _compute(batch)
    assert torch.allclose(result.advantages, torch.full((1, 1, 2), 0.5))
    assert torch.allclose(result.value_targets, torch.ones((1, 1, 2)))


def test_truncation_uses_terminal_bootstrap_but_stops_recursion() -> None:
    batch = _batch(
        rewards=[1.0, 100.0],
        values=[0.5, 0.0],
        next_values=[2.0, 0.0],
        truncated=[True, False],
        episode_starts=[False, True],
    )
    result = _compute(batch)
    assert torch.allclose(result.advantages[0], torch.full((1, 2), 2.3))
    assert torch.allclose(result.advantages[1], torch.full((1, 2), 100.0))


def test_episode_start_prevents_cross_episode_advantage_leakage() -> None:
    batch = _batch(
        rewards=[1.0, 10.0],
        values=[0.0, 0.0],
        next_values=[0.0, 0.0],
        episode_starts=[False, True],
    )
    result = _compute(batch)
    assert torch.allclose(result.advantages[0], torch.ones((1, 2)))


def test_inactive_agents_are_zero_and_do_not_affect_normalization() -> None:
    active = torch.tensor([[[True, False]], [[True, False]]])
    result = _compute(
        _batch(
            rewards=[1.0, 2.0],
            values=[0.0, 0.0],
            next_values=[0.0, 0.0],
            active_agents=active,
        ),
        normalize=True,
    )
    assert result.valid_samples.tolist() == [[[True, False]], [[True, False]]]
    assert torch.allclose(result.advantages[:, :, 1], torch.zeros((2, 1)))
    assert torch.allclose(result.advantages[result.valid_samples].mean(), torch.tensor(0.0))


def test_near_zero_advantage_variance_remains_finite() -> None:
    batch = _batch(rewards=[1.0], values=[0.0], next_values=[0.0])
    result = _compute(batch, normalize=True)
    assert torch.equal(result.advantages, torch.zeros_like(result.advantages))
    assert bool(torch.isfinite(result.advantages).all())


def test_empty_valid_samples_and_non_finite_inputs_are_rejected() -> None:
    batch = _batch(
        rewards=[1.0],
        values=[0.0],
        next_values=[0.0],
        active_agents=torch.zeros((1, 1, 2), dtype=torch.bool),
    )
    with pytest.raises(TrainingError, match="at least one valid"):
        _compute(batch)
    with pytest.raises(NumericalError, match="rewards"):
        _compute(replace(batch, rewards=torch.tensor([[float("nan")]])))


@pytest.mark.parametrize(
    ("gamma", "gae_lambda", "epsilon", "match"),
    [
        (0.0, 0.9, 1e-8, "gamma"),
        (0.9, 1.1, 1e-8, "gae_lambda"),
        (0.9, 0.8, 0.0, "epsilon"),
    ],
)
def test_gae_hyperparameters_are_validated(
    gamma: float,
    gae_lambda: float,
    epsilon: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        compute_gae(
            _batch(rewards=[1.0], values=[0.0], next_values=[0.0]),
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=False,
            normalization_epsilon=epsilon,
        )
