"""Validated configuration for the single v0.1 MAPPO-style learner."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from kovara9.config.models import StrictModel

type HiddenWidth = Annotated[int, Field(ge=8, le=4096)]
type DeviceName = Literal["auto", "cpu", "cuda"]


class NetworkConfig(StrictModel):
    """Feed-forward actor and critic network dimensions."""

    actor_hidden_sizes: tuple[HiddenWidth, ...] = Field(min_length=1, max_length=4)
    critic_hidden_sizes: tuple[HiddenWidth, ...] = Field(min_length=1, max_length=4)
    activation: Literal["relu", "tanh"] = "tanh"


class TrainingConfig(StrictModel):
    """Complete reproducible configuration for v0.1 training."""

    schema_version: int = Field(default=1, ge=1, le=1)
    algorithm: Literal["shared-actor-centralized-critic-ppo"] = (
        "shared-actor-centralized-critic-ppo"
    )
    environment_config: Path
    validation_config: Path
    network: NetworkConfig
    rollout_length: int = Field(ge=2, le=4096)
    num_environments: int = Field(ge=1, le=256)
    ppo_epochs: int = Field(ge=1, le=64)
    minibatch_size: int = Field(ge=1, le=1_000_000)
    discount_factor: float = Field(gt=0.0, le=1.0)
    gae_lambda: float = Field(ge=0.0, le=1.0)
    clipping_coefficient: float = Field(gt=0.0, le=1.0)
    entropy_coefficient: float = Field(ge=0.0, le=1.0)
    value_coefficient: float = Field(ge=0.0, le=10.0)
    maximum_gradient_norm: float = Field(gt=0.0, le=1_000.0)
    learning_rate: float = Field(gt=0.0, le=1.0)
    total_environment_steps: int = Field(ge=1)
    checkpoint_frequency: int = Field(ge=1)
    evaluation_frequency: int = Field(ge=1)
    device: DeviceName
    deterministic_torch: bool
    seed: int = Field(ge=0)

    @property
    def rollout_environment_steps(self) -> int:
        """Joint environment transitions collected by one rollout."""

        return self.rollout_length * self.num_environments

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        rollout_steps = self.rollout_environment_steps
        if self.total_environment_steps % rollout_steps != 0:
            raise ValueError(
                "total_environment_steps must be divisible by rollout_length * num_environments"
            )
        for name, frequency in (
            ("checkpoint_frequency", self.checkpoint_frequency),
            ("evaluation_frequency", self.evaluation_frequency),
        ):
            if frequency > self.total_environment_steps:
                raise ValueError(f"{name} cannot exceed total_environment_steps")
            if frequency % rollout_steps != 0:
                raise ValueError(f"{name} must be divisible by rollout_length * num_environments")
        return self
