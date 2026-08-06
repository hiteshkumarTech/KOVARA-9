"""Strict configuration schemas."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kovara9.core.types import MAX_AGENTS


class StrictModel(BaseModel):
    """Shared strict and immutable model behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a validated copy, including any requested updates.

        Pydantic's default ``model_copy(update=...)`` trusts update data. Research
        configuration must instead pass through the same validation as file input.
        """

        data = self.model_dump(mode="python", round_trip=True)
        if deep:
            data = copy.deepcopy(data)
        if update:
            data.update(update)
        return type(self).model_validate(data)


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
        maximum_step_reward = (
            self.num_targets * abs(self.reward.target_recovery)
            + abs(self.reward.success_bonus)
            + abs(self.reward.step_penalty)
            + self.num_agents * abs(self.reward.message_penalty)
        )
        if not math.isfinite(maximum_step_reward):
            raise ValueError("reward configuration can produce a non-finite team reward")
        return self


class SeedRangeConfig(StrictModel):
    """Inclusive-exclusive seed range for one scientific partition."""

    start: int = Field(ge=0)
    count: int = Field(ge=1, le=1_000_000)

    @property
    def resolved_seeds(self) -> range:
        """Return the immutable half-open range represented by this configuration."""

        return range(self.start, self.start + self.count)


class SeedPartitionsConfig(StrictModel):
    """Scientifically separate train, validation, and test seed domains."""

    train: SeedRangeConfig
    validation: SeedRangeConfig
    test: SeedRangeConfig

    @model_validator(mode="after")
    def partitions_must_not_overlap(self) -> Self:
        named_ranges = {
            "train": self.train.resolved_seeds,
            "validation": self.validation.resolved_seeds,
            "test": self.test.resolved_seeds,
        }
        names = tuple(named_ranges)
        for index, first_name in enumerate(names):
            first = named_ranges[first_name]
            for second_name in names[index + 1 :]:
                second = named_ranges[second_name]
                if max(first.start, second.start) < min(first.stop, second.stop):
                    raise ValueError(f"seed partitions overlap: {first_name} and {second_name}")
        return self

    def seeds_for(self, name: Literal["train", "validation", "test"]) -> range:
        """Return the configured range for a named partition."""

        if name == "train":
            return self.train.resolved_seeds
        if name == "validation":
            return self.validation.resolved_seeds
        return self.test.resolved_seeds


class DemoEpisodeConfig(StrictModel):
    """One explicitly seeded baseline episode in the public walkthrough."""

    name: str = Field(min_length=1, max_length=100)
    policy: Literal["random", "frontier"]
    seed: int = Field(ge=0)
    render: bool


class DemoConfig(StrictModel):
    """Bounded, non-benchmark configuration for the packaged public demo."""

    schema_version: int = Field(default=1, ge=1, le=1)
    name: str = Field(min_length=1, max_length=100)
    environment: EnvConfig
    seed_partitions: SeedPartitionsConfig
    episodes: tuple[DemoEpisodeConfig, ...] = Field(min_length=2, max_length=16)
    frame_capture_limit: int = Field(ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_walkthrough(self) -> Self:
        names = tuple(episode.name for episode in self.episodes)
        if len(set(names)) != len(names):
            raise ValueError("demo episode names must be unique")
        seeds = tuple(episode.seed for episode in self.episodes)
        if len(set(seeds)) != len(seeds):
            raise ValueError("demo episode seeds must be unique")
        policies = {episode.policy for episode in self.episodes}
        if policies != {"random", "frontier"}:
            raise ValueError("demo must include both random and frontier baseline policies")
        training_seeds = self.seed_partitions.train.resolved_seeds
        outside = [seed for seed in seeds if seed not in training_seeds]
        if outside:
            raise ValueError(f"demo seeds are outside the declared train partition: {outside}")
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

    schema_version: int = Field(default=2, ge=2, le=2)
    name: str = Field(min_length=1, max_length=100)
    seeds: tuple[int, ...] | None = None
    seed_start: int | None = Field(default=None, ge=0)
    num_episodes: int | None = Field(default=None, ge=1, le=100_000)
    bootstrap_samples: int = Field(default=2_000, ge=0, le=100_000)
    bootstrap_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed_partition: Literal["train", "validation", "test"]
    seed_partitions: SeedPartitionsConfig
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
        allowed = self.seed_partitions.seeds_for(self.seed_partition)
        outside = [seed for seed in resolved if seed not in allowed]
        if outside:
            raise ValueError(
                f"evaluation seeds are outside the {self.seed_partition} partition: {outside[:5]}"
            )
        return self

    @property
    def resolved_seeds(self) -> tuple[int, ...]:
        """Return the explicit immutable seed suite."""

        if self.seeds is not None:
            return self.seeds
        if self.seed_start is None or self.num_episodes is None:
            return ()
        return tuple(range(self.seed_start, self.seed_start + self.num_episodes))
