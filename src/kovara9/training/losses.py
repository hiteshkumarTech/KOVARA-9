"""MAPPO-style parameter-sharing PPO losses over environment-independent tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.distributions import FactoredActionDistribution
from kovara9.training.networks import ActorInput, CentralizedCritic, CriticInput, SharedActor


@dataclass(frozen=True, slots=True)
class PPOMinibatch:
    """Flat optimizer samples with separate actor and critic feature contracts."""

    actor_features: Tensor
    critic_features: Tensor
    move_action_masks: Tensor
    message_action_masks: Tensor
    move_actions: Tensor
    message_actions: Tensor
    old_joint_log_probabilities: Tensor
    advantages: Tensor
    value_targets: Tensor
    valid_samples: Tensor

    @property
    def sample_count(self) -> int:
        """Return the number of flat candidate samples."""

        return int(self.actor_features.shape[0])

    def select(self, indices: Tensor) -> PPOMinibatch:
        """Select one deterministic minibatch without changing field ordering."""

        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise TrainingError("minibatch indices must be a one-dimensional int64 tensor")
        return PPOMinibatch(
            actor_features=self.actor_features[indices],
            critic_features=self.critic_features[indices],
            move_action_masks=self.move_action_masks[indices],
            message_action_masks=self.message_action_masks[indices],
            move_actions=self.move_actions[indices],
            message_actions=self.message_actions[indices],
            old_joint_log_probabilities=self.old_joint_log_probabilities[indices],
            advantages=self.advantages[indices],
            value_targets=self.value_targets[indices],
            valid_samples=self.valid_samples[indices],
        )


@dataclass(frozen=True, slots=True)
class SurrogateDiagnostics:
    """Controlled clipped-surrogate values before model-specific losses."""

    policy_loss: Tensor
    probability_ratios: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    mean_probability_ratio: Tensor
    valid_sample_count: int


@dataclass(frozen=True, slots=True)
class PPOLossDiagnostics:
    """Differentiable losses and detached-compatible scalar diagnostics."""

    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    move_entropy: Tensor
    message_entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    mean_probability_ratio: Tensor
    probability_ratios: Tensor
    valid_sample_count: int
    explained_variance: Tensor | None


def _require_finite(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise TrainingError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise NumericalError(f"{name} contains NaN or infinite values")


def clipped_surrogate_loss(
    *,
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    valid_samples: Tensor,
    clipping_coefficient: float,
) -> SurrogateDiagnostics:
    """Compute the clipped joint-action PPO surrogate over valid samples only."""

    if not 0.0 < clipping_coefficient <= 1.0:
        raise ValueError("clipping_coefficient must be in (0, 1]")
    expected = new_log_probabilities.shape
    if not expected or old_log_probabilities.shape != expected or advantages.shape != expected:
        raise TrainingError("new log probabilities, old log probabilities, and advantages mismatch")
    if valid_samples.shape != expected or valid_samples.dtype != torch.bool:
        raise TrainingError("valid_samples must be a bool tensor matching log probabilities")
    for name, tensor in (
        ("new log probabilities", new_log_probabilities),
        ("old log probabilities", old_log_probabilities),
        ("advantages", advantages),
    ):
        _require_finite(name, tensor)
        if tensor.device != new_log_probabilities.device:
            raise TrainingError(f"{name} must use the loss device")
    if valid_samples.device != new_log_probabilities.device:
        raise TrainingError("valid_samples must use the loss device")
    if not bool(valid_samples.any()):
        raise TrainingError("PPO loss requires at least one valid sample")

    log_ratios = new_log_probabilities - old_log_probabilities
    ratios = torch.exp(log_ratios)
    _require_finite("probability ratios", ratios)
    selected_ratios = ratios[valid_samples]
    selected_log_ratios = log_ratios[valid_samples]
    selected_advantages = advantages[valid_samples]
    clipped_ratios = torch.clamp(
        selected_ratios,
        1.0 - clipping_coefficient,
        1.0 + clipping_coefficient,
    )
    surrogate = torch.minimum(
        selected_ratios * selected_advantages,
        clipped_ratios * selected_advantages,
    )
    policy_loss = -surrogate.mean()
    approximate_kl = ((selected_ratios - 1.0) - selected_log_ratios).mean()
    clip_fraction = (
        (torch.abs(selected_ratios - 1.0) > clipping_coefficient).to(torch.float32).mean()
    )
    mean_probability_ratio = selected_ratios.mean()
    for name, tensor in (
        ("policy loss", policy_loss),
        ("approximate KL", approximate_kl),
        ("clip fraction", clip_fraction),
        ("mean probability ratio", mean_probability_ratio),
    ):
        _require_finite(name, tensor)
    return SurrogateDiagnostics(
        policy_loss=policy_loss,
        probability_ratios=ratios,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        mean_probability_ratio=mean_probability_ratio,
        valid_sample_count=int(valid_samples.sum().item()),
    )


def _validate_minibatch(  # noqa: PLR0912
    batch: PPOMinibatch,
    actor: SharedActor,
    critic: CentralizedCritic,
) -> None:
    if batch.actor_features.ndim != 2 or batch.actor_features.shape[1] != actor.input_dim:
        raise TrainingError("actor_features are incompatible with the shared actor")
    sample_count = batch.sample_count
    expected_vector = (sample_count,)
    expected_move_mask = (sample_count, actor.move_action_count)
    expected_message_mask = (sample_count, actor.message_action_count)
    if batch.critic_features.shape != (sample_count, critic.input_dim):
        raise TrainingError("critic_features are incompatible with the centralized critic")
    if batch.move_action_masks.shape != expected_move_mask:
        raise TrainingError("movement masks are incompatible with the actor movement head")
    if batch.message_action_masks.shape != expected_message_mask:
        raise TrainingError("message masks are incompatible with the actor message head")
    for name in (
        "move_actions",
        "message_actions",
        "old_joint_log_probabilities",
        "advantages",
        "value_targets",
        "valid_samples",
    ):
        if getattr(batch, name).shape != expected_vector:
            raise TrainingError(f"{name} must have shape {expected_vector}")
    for name in ("move_action_masks", "message_action_masks", "valid_samples"):
        if getattr(batch, name).dtype != torch.bool:
            raise TrainingError(f"{name} must use bool dtype")
    for name in ("move_actions", "message_actions"):
        if getattr(batch, name).dtype != torch.int64:
            raise TrainingError(f"{name} must use int64 dtype")
    device = batch.actor_features.device
    for name in (
        "critic_features",
        "move_action_masks",
        "message_action_masks",
        "move_actions",
        "message_actions",
        "old_joint_log_probabilities",
        "advantages",
        "value_targets",
        "valid_samples",
    ):
        if getattr(batch, name).device != device:
            raise TrainingError(f"{name} must use the minibatch device")
    for name in (
        "actor_features",
        "critic_features",
        "old_joint_log_probabilities",
        "advantages",
        "value_targets",
    ):
        _require_finite(name, getattr(batch, name))
    if not bool(batch.valid_samples.any()):
        raise TrainingError("PPO minibatch contains zero valid samples")


def mappo_style_loss(  # noqa: PLR0913
    *,
    actor: SharedActor,
    critic: CentralizedCritic,
    batch: PPOMinibatch,
    clipping_coefficient: float,
    entropy_coefficient: float,
    value_coefficient: float,
    explained_variance_epsilon: float,
) -> PPOLossDiagnostics:
    """Evaluate decentralized actor and centralized critic losses together."""

    if entropy_coefficient < 0.0:
        raise ValueError("entropy_coefficient must be non-negative")
    if value_coefficient < 0.0:
        raise ValueError("value_coefficient must be non-negative")
    if explained_variance_epsilon <= 0.0:
        raise ValueError("explained_variance_epsilon must be positive")
    _validate_minibatch(batch, actor, critic)
    actor_logits = actor(ActorInput(batch.actor_features))
    action_statistics = FactoredActionDistribution(
        actor_logits,
        move_action_masks=batch.move_action_masks,
        message_action_masks=batch.message_action_masks,
    ).evaluate_actions(batch.move_actions, batch.message_actions)
    surrogate = clipped_surrogate_loss(
        new_log_probabilities=action_statistics.joint_log_probabilities,
        old_log_probabilities=batch.old_joint_log_probabilities,
        advantages=batch.advantages,
        valid_samples=batch.valid_samples,
        clipping_coefficient=clipping_coefficient,
    )
    predicted_values = critic(CriticInput(batch.critic_features))
    valid = batch.valid_samples
    value_residuals = predicted_values[valid] - batch.value_targets[valid]
    value_loss = 0.5 * torch.square(value_residuals).mean()
    entropy = action_statistics.joint_entropy[valid].mean()
    move_entropy = action_statistics.move_entropy[valid].mean()
    message_entropy = action_statistics.message_entropy[valid].mean()
    target_variance = batch.value_targets[valid].var(unbiased=False)
    explained_variance = (
        None
        if bool(target_variance <= explained_variance_epsilon)
        else 1.0 - value_residuals.var(unbiased=False) / target_variance
    )
    total_loss = (
        surrogate.policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    )
    for name, tensor in (
        ("value loss", value_loss),
        ("joint entropy", entropy),
        ("movement entropy", move_entropy),
        ("message entropy", message_entropy),
        ("total loss", total_loss),
    ):
        _require_finite(name, tensor)
    if explained_variance is not None:
        _require_finite("explained variance", explained_variance)
    return PPOLossDiagnostics(
        total_loss=total_loss,
        policy_loss=surrogate.policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        move_entropy=move_entropy,
        message_entropy=message_entropy,
        approximate_kl=surrogate.approximate_kl,
        clip_fraction=surrogate.clip_fraction,
        mean_probability_ratio=surrogate.mean_probability_ratio,
        probability_ratios=surrogate.probability_ratios,
        valid_sample_count=surrogate.valid_sample_count,
        explained_variance=explained_variance,
    )
