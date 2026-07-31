"""Shared actor and centralized critic network foundations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from kovara9.core.errors import NumericalError
from kovara9.training.config import NetworkConfig


@dataclass(frozen=True, slots=True)
class ActorInput:
    """Features derived only from decentralized agent observations."""

    features: Tensor


@dataclass(frozen=True, slots=True)
class CriticInput:
    """Features derived only from the centralized environment state."""

    features: Tensor


@dataclass(frozen=True, slots=True)
class ActorLogits:
    """Unnormalized logits for the two factored discrete action heads."""

    move: Tensor
    message: Tensor


@contextmanager
def _seeded_torch_initialization(seed: int) -> Iterator[None]:
    """Isolate explicitly seeded module initialization from process RNG state."""

    if seed < 0:
        raise ValueError("network seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _activation(name: Literal["relu", "tanh"]) -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.Tanh()


def _mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    activation: Literal["relu", "tanh"],
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), _activation(activation)))
        previous = width
    return nn.Sequential(*layers), previous


def _validate_features(name: str, features: Tensor, expected_dim: int) -> None:
    if features.ndim != 2 or features.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} features must have shape [batch, {expected_dim}], got {tuple(features.shape)}"
        )
    if not features.is_floating_point():
        raise TypeError(f"{name} features must use a floating-point dtype")
    if not bool(torch.isfinite(features).all()):
        raise NumericalError(f"{name} features contain NaN or infinite values")


class SharedActor(nn.Module):
    """One feed-forward actor shared across every homogeneous agent."""

    def __init__(
        self,
        *,
        input_dim: int,
        move_action_count: int,
        message_action_count: int,
        config: NetworkConfig,
        seed: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("actor input_dim must be positive")
        if move_action_count <= 0 or message_action_count <= 0:
            raise ValueError("actor action counts must be positive")
        self.input_dim = input_dim
        self.move_action_count = move_action_count
        self.message_action_count = message_action_count
        with _seeded_torch_initialization(seed):
            self.encoder, encoded_dim = _mlp(
                input_dim,
                config.actor_hidden_sizes,
                config.activation,
            )
            self.move_head = nn.Linear(encoded_dim, move_action_count)
            self.message_head = nn.Linear(encoded_dim, message_action_count)

    def forward(self, inputs: ActorInput) -> ActorLogits:
        """Return factored action logits for decentralized inputs."""

        if not isinstance(inputs, ActorInput):
            raise TypeError("SharedActor accepts ActorInput only")
        _validate_features("actor", inputs.features, self.input_dim)
        encoded = self.encoder(inputs.features)
        move = self.move_head(encoded)
        message = self.message_head(encoded)
        if not bool(torch.isfinite(move).all()) or not bool(torch.isfinite(message).all()):
            raise NumericalError("actor produced NaN or infinite logits")
        return ActorLogits(move=move, message=message)


class CentralizedCritic(nn.Module):
    """Training-only value network over centralized state features."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: NetworkConfig,
        seed: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("critic input_dim must be positive")
        self.input_dim = input_dim
        with _seeded_torch_initialization(seed):
            self.encoder, encoded_dim = _mlp(
                input_dim,
                config.critic_hidden_sizes,
                config.activation,
            )
            self.value_head = nn.Linear(encoded_dim, 1)

    def forward(self, inputs: CriticInput) -> Tensor:
        """Return one shared-team value estimate per environment state."""

        if not isinstance(inputs, CriticInput):
            raise TypeError("CentralizedCritic accepts CriticInput only")
        _validate_features("critic", inputs.features, self.input_dim)
        values = torch.squeeze(self.value_head(self.encoder(inputs.features)), dim=-1)
        if not bool(torch.isfinite(values).all()):
            raise NumericalError("critic produced NaN or infinite values")
        return values
