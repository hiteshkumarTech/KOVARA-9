"""Deterministic minibatch optimization for the v0.1 MAPPO-style learner."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import ValidationError
from torch import Tensor, nn

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.config import TrainingConfig
from kovara9.training.gae import GAEResult
from kovara9.training.losses import PPOLossDiagnostics, PPOMinibatch, mappo_style_loss
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.rollout import RolloutBatch
from kovara9.training.runtime import make_torch_generator


@dataclass(frozen=True, slots=True)
class PPOUpdateDiagnostics:
    """Aggregate scalar results and exact sample order from one PPO update."""

    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    move_entropy: float
    message_entropy: float
    approximate_kl: float
    clip_fraction: float
    mean_probability_ratio: float
    explained_variance: float | None
    maximum_pre_clip_gradient_norm: float
    maximum_post_clip_gradient_norm: float
    valid_sample_count: int
    minibatch_count: int
    epoch_sample_orders: tuple[tuple[int, ...], ...]


def _require_finite(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise TrainingError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(tensor).all()):
        raise NumericalError(f"{name} contains NaN or infinite values")


def _validate_rollout_for_optimization(batch: RolloutBatch, gae: GAEResult) -> None:
    if batch.actor_features.ndim != 4:
        raise TrainingError("actor rollout features must have shape [time, env, agent, feature]")
    time, environments, agents, _features = batch.actor_features.shape
    sample_shape = (time, environments, agents)
    environment_shape = (time, environments)
    if batch.critic_features.ndim != 3 or batch.critic_features.shape[:2] != environment_shape:
        raise TrainingError("critic rollout features must have shape [time, env, feature]")
    for name in (
        "move_actions",
        "message_actions",
        "joint_log_probabilities",
        "active_agents",
    ):
        if getattr(batch, name).shape != sample_shape:
            raise TrainingError(f"{name} must have shape {sample_shape}")
    for name in ("move_action_masks", "message_action_masks"):
        tensor = getattr(batch, name)
        if tensor.ndim != 4 or tensor.shape[:3] != sample_shape:
            raise TrainingError(f"{name} must have leading shape {sample_shape}")
    for name in ("advantages", "value_targets", "valid_samples"):
        if getattr(gae, name).shape != sample_shape:
            raise TrainingError(f"GAE {name} must have shape {sample_shape}")
    if not torch.equal(gae.valid_samples, batch.active_agents):
        raise TrainingError("GAE valid samples are incompatible with rollout active agents")
    device = batch.actor_features.device
    tensors = (
        batch.critic_features,
        batch.move_action_masks,
        batch.message_action_masks,
        batch.move_actions,
        batch.message_actions,
        batch.joint_log_probabilities,
        batch.active_agents,
        gae.advantages,
        gae.value_targets,
        gae.valid_samples,
    )
    if any(tensor.device != device for tensor in tensors):
        raise TrainingError("rollout and GAE tensors must share one device")
    for name, tensor in (
        ("actor rollout features", batch.actor_features),
        ("critic rollout features", batch.critic_features),
        ("old joint log probabilities", batch.joint_log_probabilities),
        ("advantages", gae.advantages),
        ("value targets", gae.value_targets),
    ):
        _require_finite(name, tensor)
    if not bool(gae.valid_samples.any()):
        raise TrainingError("PPO update requires at least one valid agent transition")


def build_optimization_minibatch(batch: RolloutBatch, gae: GAEResult) -> PPOMinibatch:
    """Flatten a rollout without mixing critic or actor feature contracts."""

    _validate_rollout_for_optimization(batch, gae)
    time, environments, agents, actor_features = batch.actor_features.shape
    critic_features = batch.critic_features.shape[-1]
    critic_by_agent = batch.critic_features.unsqueeze(2).expand(
        time,
        environments,
        agents,
        critic_features,
    )
    valid_indices = torch.nonzero(gae.valid_samples.reshape(-1), as_tuple=False).squeeze(-1)
    candidate = PPOMinibatch(
        actor_features=batch.actor_features.reshape(-1, actor_features),
        critic_features=critic_by_agent.reshape(-1, critic_features),
        move_action_masks=batch.move_action_masks.reshape(-1, batch.move_action_masks.shape[-1]),
        message_action_masks=batch.message_action_masks.reshape(
            -1, batch.message_action_masks.shape[-1]
        ),
        move_actions=batch.move_actions.reshape(-1),
        message_actions=batch.message_actions.reshape(-1),
        old_joint_log_probabilities=batch.joint_log_probabilities.reshape(-1),
        advantages=gae.advantages.reshape(-1),
        value_targets=gae.value_targets.reshape(-1),
        valid_samples=gae.valid_samples.reshape(-1),
    )
    selected = candidate.select(valid_indices)
    return PPOMinibatch(
        actor_features=selected.actor_features,
        critic_features=selected.critic_features,
        move_action_masks=selected.move_action_masks,
        message_action_masks=selected.message_action_masks,
        move_actions=selected.move_actions,
        message_actions=selected.message_actions,
        old_joint_log_probabilities=selected.old_joint_log_probabilities,
        advantages=selected.advantages,
        value_targets=selected.value_targets,
        valid_samples=torch.ones(
            selected.sample_count,
            dtype=torch.bool,
            device=selected.actor_features.device,
        ),
    )


def _gradient_norm(parameters: tuple[nn.Parameter, ...]) -> float:
    squared_norm = 0.0
    found_gradient = False
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        found_gradient = True
        if not bool(torch.isfinite(gradient).all()):
            raise NumericalError("PPO update produced a NaN or infinite gradient")
        squared_norm += float(torch.square(gradient.detach()).sum().item())
    if not found_gradient:
        raise TrainingError("PPO update produced no gradients")
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm):
        raise NumericalError("PPO gradient norm is NaN or infinite")
    return norm


def _finite_parameters(parameters: tuple[nn.Parameter, ...]) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


class PPOOptimizer:
    """Coordinate one Adam optimizer across a shared actor and centralized critic."""

    def __init__(
        self,
        *,
        actor: SharedActor,
        critic: CentralizedCritic,
        config: TrainingConfig,
        shuffle_seed: int,
    ) -> None:
        try:
            TrainingConfig.model_validate(config.model_dump(warnings="none"))
        except ValidationError as exc:
            raise TrainingError(
                "PPO optimizer received an invalid or unvalidated training configuration"
            ) from exc
        self.actor = actor
        self.critic = critic
        self.config = config
        actor_parameters = tuple(actor.parameters())
        critic_parameters = tuple(critic.parameters())
        self.parameters = actor_parameters + critic_parameters
        if len({id(parameter) for parameter in self.parameters}) != len(self.parameters):
            raise TrainingError("actor and critic contain duplicate optimizer parameters")
        if not self.parameters:
            raise TrainingError("PPO optimizer requires trainable actor and critic parameters")
        devices = {parameter.device for parameter in self.parameters}
        if len(devices) != 1:
            raise TrainingError("actor and critic parameters must share one device")
        self.device = next(iter(devices))
        if not _finite_parameters(self.parameters):
            raise NumericalError("PPO optimizer received non-finite model parameters")
        self.shuffle_generator = make_torch_generator(shuffle_seed, self.device)
        self.optimizer = torch.optim.Adam(self.parameters, lr=config.learning_rate)

    def update(self, batch: RolloutBatch, gae: GAEResult) -> PPOUpdateDiagnostics:
        """Run configured deterministic PPO epochs over every valid transition."""

        optimization_batch = build_optimization_minibatch(batch, gae)
        valid_count = optimization_batch.sample_count
        if valid_count <= 0:
            raise TrainingError("PPO update received an empty optimization batch")
        self.actor.train()
        self.critic.train()
        diagnostics: list[PPOLossDiagnostics] = []
        pre_clip_norms: list[float] = []
        post_clip_norms: list[float] = []
        epoch_orders: list[tuple[int, ...]] = []

        for _epoch in range(self.config.ppo_epochs):
            permutation = torch.randperm(
                valid_count,
                generator=self.shuffle_generator,
                device=self.device,
            )
            epoch_orders.append(tuple(int(index) for index in permutation.cpu().tolist()))
            for start in range(0, valid_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                minibatch = optimization_batch.select(indices)
                self.optimizer.zero_grad(set_to_none=True)
                loss = mappo_style_loss(
                    actor=self.actor,
                    critic=self.critic,
                    batch=minibatch,
                    clipping_coefficient=self.config.clipping_coefficient,
                    entropy_coefficient=self.config.entropy_coefficient,
                    value_coefficient=self.config.value_coefficient,
                    explained_variance_epsilon=self.config.advantage_normalization_epsilon,
                )
                if not bool(torch.isfinite(loss.total_loss)):
                    raise NumericalError("PPO total loss is NaN or infinite")
                loss.total_loss.backward()  # type: ignore[no-untyped-call]
                pre_clip_norms.append(_gradient_norm(self.parameters))
                torch.nn.utils.clip_grad_norm_(
                    self.parameters,
                    self.config.maximum_gradient_norm,
                    foreach=False,
                )
                post_clip_norm = _gradient_norm(self.parameters)
                tolerance = self.config.maximum_gradient_norm * math.sqrt(
                    torch.finfo(torch.float32).eps
                )
                if post_clip_norm > self.config.maximum_gradient_norm + tolerance:
                    raise NumericalError(
                        "gradient clipping failed to enforce maximum_gradient_norm"
                    )
                post_clip_norms.append(post_clip_norm)
                self.optimizer.step()
                if not _finite_parameters(self.parameters):
                    raise NumericalError("PPO update produced non-finite model parameters")
                diagnostics.append(loss)

        if not diagnostics:
            raise TrainingError("PPO update did not execute any minibatches")
        return _aggregate_diagnostics(
            diagnostics,
            pre_clip_norms,
            post_clip_norms,
            valid_count,
            tuple(epoch_orders),
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Export optimizer slots and the explicit minibatch shuffle stream."""

        return {
            "schema_version": 1,
            "optimizer": self.optimizer.state_dict(),
            "shuffle_generator_state": self.shuffle_generator.get_state().cpu(),
        }

    def restore_checkpoint_state(self, raw_checkpoint: Mapping[str, Any]) -> None:
        """Restore optimizer and shuffle state after configuration validation."""

        expected_keys = {"schema_version", "optimizer", "shuffle_generator_state"}
        if set(raw_checkpoint) != expected_keys or raw_checkpoint.get("schema_version") != 1:
            raise TrainingError("optimizer checkpoint fields do not match schema version 1")
        optimizer_state = raw_checkpoint["optimizer"]
        if not isinstance(optimizer_state, Mapping):
            raise TrainingError("optimizer checkpoint state must be a mapping")
        generator_state = raw_checkpoint["shuffle_generator_state"]
        if (
            not isinstance(generator_state, Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
        ):
            raise TrainingError("optimizer checkpoint shuffle RNG state is invalid")
        try:
            self.optimizer.load_state_dict(dict(optimizer_state))
            self.shuffle_generator.set_state(generator_state.cpu())
        except (RuntimeError, ValueError) as exc:
            raise TrainingError("cannot restore optimizer checkpoint state") from exc
        if not _finite_parameters(self.parameters):
            raise NumericalError("optimizer checkpoint restored non-finite model parameters")
        for state in self.optimizer.state.values():
            for value in state.values():
                if (
                    isinstance(value, Tensor)
                    and value.is_floating_point()
                    and not bool(torch.isfinite(value).all())
                ):
                    raise NumericalError("optimizer checkpoint contains non-finite slot state")


def _aggregate_diagnostics(
    diagnostics: list[PPOLossDiagnostics],
    pre_clip_norms: list[float],
    post_clip_norms: list[float],
    valid_sample_count: int,
    epoch_orders: tuple[tuple[int, ...], ...],
) -> PPOUpdateDiagnostics:
    minibatch_count = len(diagnostics)
    diagnostic_weight = sum(item.valid_sample_count for item in diagnostics)

    def mean_tensor(name: str) -> float:
        value = sum(
            float(getattr(item, name).detach().item()) * item.valid_sample_count
            for item in diagnostics
        )
        return value / diagnostic_weight

    explained = [
        (float(item.explained_variance.detach().item()), item.valid_sample_count)
        for item in diagnostics
        if item.explained_variance is not None
    ]
    return PPOUpdateDiagnostics(
        total_loss=mean_tensor("total_loss"),
        policy_loss=mean_tensor("policy_loss"),
        value_loss=mean_tensor("value_loss"),
        entropy=mean_tensor("entropy"),
        move_entropy=mean_tensor("move_entropy"),
        message_entropy=mean_tensor("message_entropy"),
        approximate_kl=mean_tensor("approximate_kl"),
        clip_fraction=mean_tensor("clip_fraction"),
        mean_probability_ratio=mean_tensor("mean_probability_ratio"),
        explained_variance=(
            sum(value * weight for value, weight in explained)
            / sum(weight for _value, weight in explained)
            if explained
            else None
        ),
        maximum_pre_clip_gradient_norm=max(pre_clip_norms),
        maximum_post_clip_gradient_norm=max(post_clip_norms),
        valid_sample_count=valid_sample_count,
        minibatch_count=minibatch_count,
        epoch_sample_orders=epoch_orders,
    )
