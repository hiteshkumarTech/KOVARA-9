"""Typed fixed-shape storage for synchronous multi-agent rollouts."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from kovara9.core.errors import NumericalError, TrainingError


@dataclass(frozen=True, slots=True)
class RolloutSpec:
    """All dimensions needed to allocate one rollout."""

    rollout_length: int
    num_environments: int
    num_agents: int
    actor_feature_dim: int
    critic_feature_dim: int
    move_action_count: int
    message_action_count: int

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) <= 0:
                raise ValueError(f"{field.name} must be positive")


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """One synchronous transition across all environments and agents."""

    actor_features: Tensor
    critic_features: Tensor
    move_action_masks: Tensor
    message_action_masks: Tensor
    move_actions: Tensor
    message_actions: Tensor
    move_log_probabilities: Tensor
    message_log_probabilities: Tensor
    rewards: Tensor
    values: Tensor
    terminated: Tensor
    truncated: Tensor
    active_agents: Tensor


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One complete immutable-by-convention rollout plus bootstrap values."""

    actor_features: Tensor
    critic_features: Tensor
    move_action_masks: Tensor
    message_action_masks: Tensor
    move_actions: Tensor
    message_actions: Tensor
    move_log_probabilities: Tensor
    message_log_probabilities: Tensor
    rewards: Tensor
    values: Tensor
    terminated: Tensor
    truncated: Tensor
    active_agents: Tensor
    next_values: Tensor


class RolloutBuffer:
    """Preallocated storage that rejects incomplete or non-finite rollouts."""

    def __init__(self, spec: RolloutSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        time = spec.rollout_length
        environments = spec.num_environments
        agents = spec.num_agents
        self.actor_features = torch.zeros(
            (time, environments, agents, spec.actor_feature_dim),
            dtype=torch.float32,
            device=device,
        )
        self.critic_features = torch.zeros(
            (time, environments, spec.critic_feature_dim),
            dtype=torch.float32,
            device=device,
        )
        self.move_action_masks = torch.zeros(
            (time, environments, agents, spec.move_action_count),
            dtype=torch.bool,
            device=device,
        )
        self.message_action_masks = torch.zeros(
            (time, environments, agents, spec.message_action_count),
            dtype=torch.bool,
            device=device,
        )
        agent_shape = (time, environments, agents)
        environment_shape = (time, environments)
        self.move_actions = torch.zeros(agent_shape, dtype=torch.int64, device=device)
        self.message_actions = torch.zeros(agent_shape, dtype=torch.int64, device=device)
        self.move_log_probabilities = torch.zeros(agent_shape, dtype=torch.float32, device=device)
        self.message_log_probabilities = torch.zeros(
            agent_shape, dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(environment_shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(environment_shape, dtype=torch.float32, device=device)
        self.terminated = torch.zeros(environment_shape, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(environment_shape, dtype=torch.bool, device=device)
        self.active_agents = torch.zeros(agent_shape, dtype=torch.bool, device=device)
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size == self.spec.rollout_length

    def append(self, step: RolloutStep) -> None:
        """Validate and append one synchronous transition."""

        if self.full:
            raise TrainingError("cannot append to a full rollout buffer")
        if bool(torch.logical_and(step.terminated, step.truncated).any()):
            raise TrainingError("a transition cannot be both terminated and truncated")
        index = self._size
        for name in (
            "actor_features",
            "critic_features",
            "move_action_masks",
            "message_action_masks",
            "move_actions",
            "message_actions",
            "move_log_probabilities",
            "message_log_probabilities",
            "rewards",
            "values",
            "terminated",
            "truncated",
            "active_agents",
        ):
            destination = getattr(self, name)[index]
            source = getattr(step, name)
            if source.shape != destination.shape:
                raise TrainingError(
                    f"{name} must have shape {tuple(destination.shape)}, got {tuple(source.shape)}"
                )
            converted = source.to(device=self.device, dtype=destination.dtype)
            if converted.is_floating_point() and not bool(torch.isfinite(converted).all()):
                raise NumericalError(f"{name} contains NaN or infinite values")
            destination.copy_(converted)
        self._size += 1

    def as_batch(self, next_values: Tensor) -> RolloutBatch:
        """Return a complete rollout after validating critic bootstrap values."""

        if not self.full:
            raise TrainingError(
                f"rollout buffer is incomplete: {self._size}/{self.spec.rollout_length} steps"
            )
        expected_shape = (self.spec.num_environments,)
        if next_values.shape != expected_shape:
            raise TrainingError(
                f"next_values must have shape {expected_shape}, got {tuple(next_values.shape)}"
            )
        converted_next_values = next_values.to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(converted_next_values).all()):
            raise NumericalError("next_values contains NaN or infinite values")
        return RolloutBatch(
            actor_features=self.actor_features.clone(),
            critic_features=self.critic_features.clone(),
            move_action_masks=self.move_action_masks.clone(),
            message_action_masks=self.message_action_masks.clone(),
            move_actions=self.move_actions.clone(),
            message_actions=self.message_actions.clone(),
            move_log_probabilities=self.move_log_probabilities.clone(),
            message_log_probabilities=self.message_log_probabilities.clone(),
            rewards=self.rewards.clone(),
            values=self.values.clone(),
            terminated=self.terminated.clone(),
            truncated=self.truncated.clone(),
            active_agents=self.active_agents.clone(),
            next_values=converted_next_values.clone(),
        )

    def reset(self) -> None:
        """Make the fixed allocation available for the next complete rollout."""

        self._size = 0
