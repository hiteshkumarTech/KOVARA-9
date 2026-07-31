"""Typed fixed-shape storage for synchronous multi-agent rollouts."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.core.types import AgentId


@dataclass(frozen=True, slots=True)
class RolloutSpec:
    """All dimensions needed to allocate one rollout."""

    rollout_length: int
    num_environments: int
    agent_order: tuple[AgentId, ...]
    actor_feature_dim: int
    critic_feature_dim: int
    move_action_count: int
    message_action_count: int

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name == "agent_order":
                continue
            if getattr(self, field.name) <= 0:
                raise ValueError(f"{field.name} must be positive")
        if not self.agent_order:
            raise ValueError("agent_order cannot be empty")
        if len(set(self.agent_order)) != len(self.agent_order):
            raise ValueError("agent_order must contain unique agent identifiers")

    @property
    def num_agents(self) -> int:
        """Return the fixed homogeneous agent-slot count."""

        return len(self.agent_order)


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
    joint_log_probabilities: Tensor
    rewards: Tensor
    values: Tensor
    next_values: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_starts: Tensor
    active_agents: Tensor
    communication_rejections: Tensor
    environment_ids: Tensor
    transition_ids: Tensor


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
    joint_log_probabilities: Tensor
    rewards: Tensor
    values: Tensor
    next_values: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_starts: Tensor
    active_agents: Tensor
    communication_rejections: Tensor
    environment_ids: Tensor
    transition_ids: Tensor
    agent_order: tuple[AgentId, ...]


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
        self.joint_log_probabilities = torch.zeros(agent_shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(environment_shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(environment_shape, dtype=torch.float32, device=device)
        self.next_values = torch.zeros(environment_shape, dtype=torch.float32, device=device)
        self.terminated = torch.zeros(environment_shape, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(environment_shape, dtype=torch.bool, device=device)
        self.episode_starts = torch.zeros(environment_shape, dtype=torch.bool, device=device)
        self.active_agents = torch.zeros(agent_shape, dtype=torch.bool, device=device)
        self.communication_rejections = torch.zeros(agent_shape, dtype=torch.bool, device=device)
        self.environment_ids = torch.zeros(environment_shape, dtype=torch.int64, device=device)
        self.transition_ids = torch.zeros(environment_shape, dtype=torch.int64, device=device)
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
            "joint_log_probabilities",
            "rewards",
            "values",
            "next_values",
            "terminated",
            "truncated",
            "episode_starts",
            "active_agents",
            "communication_rejections",
            "environment_ids",
            "transition_ids",
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

    def as_batch(self) -> RolloutBatch:
        """Return a complete rollout with per-transition bootstrap values."""

        if not self.full:
            raise TrainingError(
                f"rollout buffer is incomplete: {self._size}/{self.spec.rollout_length} steps"
            )
        return RolloutBatch(
            actor_features=self.actor_features.clone(),
            critic_features=self.critic_features.clone(),
            move_action_masks=self.move_action_masks.clone(),
            message_action_masks=self.message_action_masks.clone(),
            move_actions=self.move_actions.clone(),
            message_actions=self.message_actions.clone(),
            move_log_probabilities=self.move_log_probabilities.clone(),
            message_log_probabilities=self.message_log_probabilities.clone(),
            joint_log_probabilities=self.joint_log_probabilities.clone(),
            rewards=self.rewards.clone(),
            values=self.values.clone(),
            next_values=self.next_values.clone(),
            terminated=self.terminated.clone(),
            truncated=self.truncated.clone(),
            episode_starts=self.episode_starts.clone(),
            active_agents=self.active_agents.clone(),
            communication_rejections=self.communication_rejections.clone(),
            environment_ids=self.environment_ids.clone(),
            transition_ids=self.transition_ids.clone(),
            agent_order=self.spec.agent_order,
        )

    def reset(self) -> None:
        """Make the fixed allocation available for the next complete rollout."""

        self._size = 0
