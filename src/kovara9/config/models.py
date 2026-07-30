"""Strict configuration schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kovara9.core.types import MAX_AGENTS


class StrictModel(BaseModel):
    """Shared strict and immutable model behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CommunicationConfig(StrictModel):
    """Limited discrete broadcast channel configuration."""

    enabled: bool = True
    vocabulary_size: int = Field(default=4, ge=1, le=32)
    budget_per_agent: int = Field(default=8, ge=0, le=10_000)

    @model_validator(mode="after")
    def disabled_has_zero_budget(self) -> Self:
        if not self.enabled and self.budget_per_agent != 0:
            raise ValueError("budget_per_agent must be 0 when communication is disabled")
        return self


class GenerationConfig(StrictModel):
    """Procedural generation safety limits."""

    max_attempts: int = Field(default=128, ge=1, le=10_000)


class RewardConfig(StrictModel):
    """Named components of the shared team reward."""

    target_recovery: float = 1.0
    success_bonus: float = 5.0
    step_penalty: float = -0.01
    message_penalty: float = -0.001


class EnvConfig(StrictModel):
    """Complete configuration for a grid-rescue environment instance."""

    schema_version: int = Field(default=1, ge=1, le=1)
    environment_id: str = "KovaraGridRescue-v0"
    width: int = Field(ge=5, le=64)
    height: int = Field(ge=5, le=64)
    num_agents: int = Field(ge=2, le=MAX_AGENTS)
    obstacle_density: float = Field(ge=0.0, le=0.45)
    observation_radius: int = Field(ge=1, le=16)
    num_targets: int = Field(ge=1, le=64)
    max_steps: int = Field(ge=1, le=100_000)
    communication: CommunicationConfig = CommunicationConfig()
    generation: GenerationConfig = GenerationConfig()
    reward: RewardConfig = RewardConfig()

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.environment_id != "KovaraGridRescue-v0":
            raise ValueError(f"unsupported environment_id: {self.environment_id}")
        available_before_obstacles = self.width * self.height
        required = self.num_agents + self.num_targets
        expected_free = available_before_obstacles - round(
            available_before_obstacles * self.obstacle_density
        )
        if expected_free < required:
            raise ValueError(
                f"grid expects {expected_free} free cells but requires at least {required}"
            )
        return self


class ComparisonConfig(StrictModel):
    """Two environment configurations used for a structural comparison."""

    reference_environment: Path
    held_out_environment: Path

    @model_validator(mode="after")
    def paths_must_differ(self) -> Self:
        if self.reference_environment == self.held_out_environment:
            raise ValueError("reference and held-out environment paths must differ")
        return self


class EvaluationConfig(StrictModel):
    """Finite, reproducible evaluation suite."""

    schema_version: int = Field(default=1, ge=1, le=1)
    name: str = Field(min_length=1, max_length=100)
    seeds: tuple[int, ...] | None = None
    seed_start: int | None = Field(default=None, ge=0)
    num_episodes: int | None = Field(default=None, ge=1, le=100_000)
    bootstrap_samples: int = Field(default=2_000, ge=0, le=100_000)
    bootstrap_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    comparison: ComparisonConfig | None = None

    @model_validator(mode="after")
    def validate_seed_source(self) -> Self:
        has_explicit = self.seeds is not None
        has_range = self.seed_start is not None or self.num_episodes is not None
        if has_explicit == has_range:
            raise ValueError("provide exactly one of seeds or seed_start plus num_episodes")
        if has_range and (self.seed_start is None or self.num_episodes is None):
            raise ValueError("seed_start and num_episodes must be provided together")
        resolved = self.resolved_seeds
        if any(seed < 0 for seed in resolved):
            raise ValueError("evaluation seeds must be non-negative")
        if len(set(resolved)) != len(resolved):
            raise ValueError("evaluation seeds must be unique")
        return self

    @property
    def resolved_seeds(self) -> tuple[int, ...]:
        """Return the explicit immutable seed suite."""

        if self.seeds is not None:
            return self.seeds
        if self.seed_start is None or self.num_episodes is None:
            return ()
        return tuple(range(self.seed_start, self.seed_start + self.num_episodes))
