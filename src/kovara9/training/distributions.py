"""Numerically stable factored categorical action distributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.networks import ActorLogits


@dataclass(frozen=True, slots=True)
class FactoredProbabilities:
    """Exact probabilities for independently masked action factors."""

    move: Tensor
    message: Tensor


@dataclass(frozen=True, slots=True)
class FactoredActionStatistics:
    """Actions and the policy statistics needed by an on-policy rollout."""

    move_actions: Tensor
    message_actions: Tensor
    joint_log_probabilities: Tensor
    move_log_probabilities: Tensor
    message_log_probabilities: Tensor
    joint_entropy: Tensor
    move_entropy: Tensor
    message_entropy: Tensor


@dataclass(frozen=True, slots=True)
class _MaskedFactor:
    probabilities: Tensor
    log_probabilities: Tensor
    entropy: Tensor
    mask: Tensor


def _masked_factor(name: str, logits: Tensor, mask: Tensor) -> _MaskedFactor:
    if logits.ndim < 2:
        raise TrainingError(f"{name} logits must include batch and action dimensions")
    if not logits.is_floating_point():
        raise TrainingError(f"{name} logits must use a floating-point dtype")
    if mask.dtype != torch.bool:
        raise TrainingError(f"{name} action mask must use bool dtype")
    if mask.shape != logits.shape:
        raise TrainingError(
            f"{name} action mask must have shape {tuple(logits.shape)}, got {tuple(mask.shape)}"
        )
    if mask.device != logits.device:
        raise TrainingError(f"{name} logits and action mask must use the same device")
    if not bool(torch.isfinite(logits).all()):
        raise NumericalError(f"{name} logits contain NaN or infinite values")
    if not bool(mask.any(dim=-1).all()):
        raise TrainingError(f"{name} action mask has a row with no valid action")

    masked_logits = logits.masked_fill(~mask, -torch.inf)
    log_normalizer = torch.logsumexp(masked_logits, dim=-1, keepdim=True)
    log_probabilities = masked_logits - log_normalizer
    probabilities = torch.exp(log_probabilities)
    entropy_terms = torch.where(mask, probabilities * log_probabilities, 0.0)
    entropy = -entropy_terms.sum(dim=-1)
    if not bool(torch.isfinite(probabilities).all()) or not bool(torch.isfinite(entropy).all()):
        raise NumericalError(f"{name} masked categorical produced non-finite statistics")
    return _MaskedFactor(probabilities, log_probabilities, entropy, mask)


def _validated_factor_actions(name: str, factor: _MaskedFactor, actions: Tensor) -> Tensor:
    expected_shape = factor.probabilities.shape[:-1]
    if actions.shape != expected_shape:
        raise TrainingError(
            f"{name} actions must have shape {tuple(expected_shape)}, got {tuple(actions.shape)}"
        )
    if actions.dtype != torch.int64:
        raise TrainingError(f"{name} actions must use int64 dtype")
    if actions.device != factor.probabilities.device:
        raise TrainingError(f"{name} actions and probabilities must use the same device")
    action_count = factor.probabilities.shape[-1]
    if bool(((actions < 0) | (actions >= action_count)).any()):
        raise TrainingError(f"{name} actions contain an out-of-range value")
    if not bool(torch.gather(factor.mask, -1, actions.unsqueeze(-1)).all()):
        raise TrainingError(f"{name} actions contain an action rejected by the mask")
    return actions


def _sample_factor(factor: _MaskedFactor, generator: torch.Generator) -> Tensor:
    action_count = factor.probabilities.shape[-1]
    flat = factor.probabilities.reshape(-1, action_count)
    sampled = torch.multinomial(flat, 1, replacement=True, generator=generator)
    return sampled.reshape(factor.probabilities.shape[:-1])


class FactoredActionDistribution:
    """Two independent masked categorical factors with joint policy statistics."""

    def __init__(
        self,
        logits: ActorLogits,
        *,
        move_action_masks: Tensor,
        message_action_masks: Tensor,
    ) -> None:
        self._move = _masked_factor("move", logits.move, move_action_masks)
        self._message = _masked_factor("message", logits.message, message_action_masks)
        if self._move.probabilities.shape[:-1] != self._message.probabilities.shape[:-1]:
            raise TrainingError("move and message factors must have identical batch dimensions")

    @property
    def probabilities(self) -> FactoredProbabilities:
        """Return factor probabilities; invalid entries are exactly zero."""

        return FactoredProbabilities(self._move.probabilities, self._message.probabilities)

    def sample(self, *, generator: torch.Generator) -> FactoredActionStatistics:
        """Sample both factors using only the supplied generator."""

        move = _sample_factor(self._move, generator)
        message = _sample_factor(self._message, generator)
        return self.evaluate_actions(move, message)

    def mode(self) -> FactoredActionStatistics:
        """Choose each factor's masked argmax without consuming RNG state."""

        move = self._move.probabilities.argmax(dim=-1)
        message = self._message.probabilities.argmax(dim=-1)
        return self.evaluate_actions(move, message)

    def evaluate_actions(
        self,
        move_actions: Tensor,
        message_actions: Tensor,
    ) -> FactoredActionStatistics:
        """Validate supplied actions and calculate factor and joint statistics."""

        move = _validated_factor_actions("move", self._move, move_actions)
        message = _validated_factor_actions("message", self._message, message_actions)
        move_log_probabilities = torch.gather(
            self._move.log_probabilities, -1, move.unsqueeze(-1)
        ).squeeze(-1)
        message_log_probabilities = torch.gather(
            self._message.log_probabilities, -1, message.unsqueeze(-1)
        ).squeeze(-1)
        return FactoredActionStatistics(
            move_actions=move,
            message_actions=message,
            joint_log_probabilities=move_log_probabilities + message_log_probabilities,
            move_log_probabilities=move_log_probabilities,
            message_log_probabilities=message_log_probabilities,
            joint_entropy=self._move.entropy + self._message.entropy,
            move_entropy=self._move.entropy,
            message_entropy=self._message.entropy,
        )
