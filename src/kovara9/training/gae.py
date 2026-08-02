"""Generalized Advantage Estimation over explicit environment-agent axes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.rollout import RolloutBatch


@dataclass(frozen=True, slots=True)
class GAEResult:
    """Per-agent advantages, critic targets, and valid optimization samples."""

    advantages: Tensor
    unnormalized_advantages: Tensor
    value_targets: Tensor
    valid_samples: Tensor


def _validate_gae_inputs(batch: RolloutBatch) -> tuple[int, int, int]:
    if batch.active_agents.ndim != 3:
        raise TrainingError("active_agents must have shape [time, environments, agents]")
    time, environments, agents = batch.active_agents.shape
    if time <= 0 or environments <= 0 or agents <= 0:
        raise TrainingError("GAE requires non-empty time, environment, and agent dimensions")
    environment_shape = (time, environments)
    for name in (
        "rewards",
        "values",
        "next_values",
        "terminated",
        "truncated",
        "episode_starts",
    ):
        tensor = getattr(batch, name)
        if tensor.shape != environment_shape:
            raise TrainingError(
                f"{name} must have shape {environment_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.device != batch.active_agents.device:
            raise TrainingError(f"{name} must use the rollout device")
    for name in ("terminated", "truncated", "episode_starts", "active_agents"):
        if getattr(batch, name).dtype != torch.bool:
            raise TrainingError(f"{name} must use bool dtype")
    for name in ("rewards", "values", "next_values"):
        tensor = getattr(batch, name)
        if not tensor.is_floating_point():
            raise TrainingError(f"{name} must use a floating-point dtype")
        if not bool(torch.isfinite(tensor).all()):
            raise NumericalError(f"{name} contains NaN or infinite values")
    if bool(torch.logical_and(batch.terminated, batch.truncated).any()):
        raise TrainingError("a transition cannot be both terminated and truncated")
    return time, environments, agents


def _normalize_valid_advantages(
    advantages: Tensor,
    valid_samples: Tensor,
    epsilon: float,
) -> Tensor:
    valid_advantages = advantages[valid_samples]
    if valid_advantages.numel() == 0:
        raise TrainingError("advantage normalization requires at least one valid sample")
    mean = valid_advantages.mean()
    variance = valid_advantages.var(unbiased=False)
    normalized = (valid_advantages - mean) / torch.sqrt(variance + epsilon)
    if not bool(torch.isfinite(normalized).all()):
        raise NumericalError("advantage normalization produced NaN or infinite values")
    result = torch.zeros_like(advantages)
    result[valid_samples] = normalized
    return result


def compute_gae(
    batch: RolloutBatch,
    *,
    gamma: float,
    gae_lambda: float,
    normalize_advantages: bool,
    normalization_epsilon: float,
) -> GAEResult:
    """Compute GAE without crossing environment, episode, or inactive-agent boundaries."""

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be positive")
    time, environments, agents = _validate_gae_inputs(batch)
    valid_samples = batch.active_agents
    if not bool(valid_samples.any()):
        raise TrainingError("GAE requires at least one valid agent transition")

    values = batch.values.unsqueeze(-1).expand(time, environments, agents)
    next_values = batch.next_values.unsqueeze(-1).expand(time, environments, agents)
    rewards = batch.rewards.unsqueeze(-1).expand(time, environments, agents)
    terminations = batch.terminated.unsqueeze(-1).expand(time, environments, agents)
    bootstraps = torch.where(terminations, torch.zeros_like(next_values), next_values)
    deltas = rewards + gamma * bootstraps - values

    advantages = torch.zeros_like(deltas)
    next_advantage = torch.zeros((environments, agents), dtype=deltas.dtype, device=deltas.device)
    for step in reversed(range(time)):
        if step + 1 < time:
            same_episode = ~batch.episode_starts[step + 1].unsqueeze(-1)
            next_agents_valid = valid_samples[step + 1]
        else:
            same_episode = torch.zeros((environments, 1), dtype=torch.bool, device=deltas.device)
            next_agents_valid = torch.zeros(
                (environments, agents), dtype=torch.bool, device=deltas.device
            )
        boundary = torch.logical_or(batch.terminated[step], batch.truncated[step]).unsqueeze(-1)
        continues = ~boundary & same_episode & next_agents_valid
        estimate = deltas[step] + gamma * gae_lambda * continues * next_advantage
        next_advantage = torch.where(valid_samples[step], estimate, 0.0)
        advantages[step] = next_advantage

    if not bool(torch.isfinite(advantages).all()):
        raise NumericalError("GAE produced NaN or infinite advantages")
    value_targets = torch.where(valid_samples, advantages + values, 0.0)
    output_advantages = (
        _normalize_valid_advantages(advantages, valid_samples, normalization_epsilon)
        if normalize_advantages
        else advantages.clone()
    )
    return GAEResult(
        advantages=output_advantages,
        unnormalized_advantages=advantages,
        value_targets=value_targets,
        valid_samples=valid_samples.clone(),
    )
