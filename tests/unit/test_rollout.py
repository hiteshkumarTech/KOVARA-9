import pytest
import torch

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.rollout import RolloutBuffer, RolloutSpec, RolloutStep


def _spec() -> RolloutSpec:
    return RolloutSpec(
        rollout_length=2,
        num_environments=2,
        agent_order=("agent_0", "agent_1", "agent_2"),
        actor_feature_dim=4,
        critic_feature_dim=5,
        move_action_count=5,
        message_action_count=2,
    )


def _step(*, terminal_environment: int | None = None) -> RolloutStep:
    terminated = torch.zeros(2, dtype=torch.bool)
    if terminal_environment is not None:
        terminated[terminal_environment] = True
    return RolloutStep(
        actor_features=torch.ones((2, 3, 4)),
        critic_features=torch.ones((2, 5)),
        move_action_masks=torch.ones((2, 3, 5), dtype=torch.bool),
        message_action_masks=torch.ones((2, 3, 2), dtype=torch.bool),
        move_actions=torch.zeros((2, 3), dtype=torch.int64),
        message_actions=torch.zeros((2, 3), dtype=torch.int64),
        move_log_probabilities=torch.full((2, 3), -0.5),
        message_log_probabilities=torch.full((2, 3), -0.25),
        joint_log_probabilities=torch.full((2, 3), -0.75),
        rewards=torch.tensor([1.0, 2.0]),
        values=torch.tensor([0.5, 0.75]),
        next_values=torch.tensor([0.25, 0.0]),
        terminated=terminated,
        truncated=torch.zeros(2, dtype=torch.bool),
        episode_starts=torch.tensor([False, True]),
        active_agents=torch.ones((2, 3), dtype=torch.bool),
        communication_rejections=torch.zeros((2, 3), dtype=torch.bool),
        environment_ids=torch.tensor([0, 1], dtype=torch.int64),
        transition_ids=torch.tensor([3, 7], dtype=torch.int64),
    )


def test_rollout_storage_shapes_and_episode_boundaries() -> None:
    buffer = RolloutBuffer(_spec(), torch.device("cpu"))
    buffer.append(_step())
    buffer.append(_step(terminal_environment=1))
    batch = buffer.as_batch()
    assert buffer.full
    assert batch.actor_features.shape == (2, 2, 3, 4)
    assert batch.critic_features.shape == (2, 2, 5)
    assert batch.move_actions.shape == (2, 2, 3)
    assert batch.rewards.shape == (2, 2)
    assert batch.terminated[:, 1].tolist() == [False, True]
    assert batch.next_values.tolist() == [[0.25, 0.0], [0.25, 0.0]]
    assert batch.episode_starts[:, 1].tolist() == [True, True]
    assert batch.environment_ids.tolist() == [[0, 1], [0, 1]]
    assert batch.transition_ids.tolist() == [[3, 7], [3, 7]]
    assert batch.agent_order == ("agent_0", "agent_1", "agent_2")
    buffer.reset()
    assert buffer.size == 0


def test_rollout_rejects_incomplete_batch_and_invalid_shape() -> None:
    buffer = RolloutBuffer(_spec(), torch.device("cpu"))
    with pytest.raises(TrainingError, match="incomplete"):
        buffer.as_batch()
    step = _step()
    invalid = RolloutStep(
        actor_features=torch.zeros((2, 3, 3)),
        critic_features=step.critic_features,
        move_action_masks=step.move_action_masks,
        message_action_masks=step.message_action_masks,
        move_actions=step.move_actions,
        message_actions=step.message_actions,
        move_log_probabilities=step.move_log_probabilities,
        message_log_probabilities=step.message_log_probabilities,
        joint_log_probabilities=step.joint_log_probabilities,
        rewards=step.rewards,
        values=step.values,
        next_values=step.next_values,
        terminated=step.terminated,
        truncated=step.truncated,
        episode_starts=step.episode_starts,
        active_agents=step.active_agents,
        communication_rejections=step.communication_rejections,
        environment_ids=step.environment_ids,
        transition_ids=step.transition_ids,
    )
    with pytest.raises(TrainingError, match="actor_features must have shape"):
        buffer.append(invalid)


def test_rollout_rejects_non_finite_values_and_double_episode_end() -> None:
    buffer = RolloutBuffer(_spec(), torch.device("cpu"))
    step = _step()
    non_finite = RolloutStep(
        actor_features=step.actor_features,
        critic_features=step.critic_features,
        move_action_masks=step.move_action_masks,
        message_action_masks=step.message_action_masks,
        move_actions=step.move_actions,
        message_actions=step.message_actions,
        move_log_probabilities=step.move_log_probabilities,
        message_log_probabilities=step.message_log_probabilities,
        joint_log_probabilities=step.joint_log_probabilities,
        rewards=torch.tensor([float("inf"), 0.0]),
        values=step.values,
        next_values=step.next_values,
        terminated=step.terminated,
        truncated=step.truncated,
        episode_starts=step.episode_starts,
        active_agents=step.active_agents,
        communication_rejections=step.communication_rejections,
        environment_ids=step.environment_ids,
        transition_ids=step.transition_ids,
    )
    with pytest.raises(NumericalError, match="rewards"):
        buffer.append(non_finite)

    double_end = _step(terminal_environment=0)
    double_end.truncated[0] = True
    with pytest.raises(TrainingError, match="both terminated and truncated"):
        buffer.append(double_end)
