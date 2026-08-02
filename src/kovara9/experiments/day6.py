"""Controlled Day 6 multi-seed experiment contracts and statistics."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from kovara9.config.loader import (
    TrainingInputs,
    configuration_fingerprint,
    load_training_inputs,
)
from kovara9.config.models import EnvConfig, EvaluationConfig, StrictModel
from kovara9.core.errors import ConfigurationError, TrainingError
from kovara9.training.checkpoint import training_definition_fingerprint
from kovara9.training.config import TrainingConfig

DAY6_ROOT_SEEDS = (0, 1, 2)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class MetricStatistics(StrictModel):
    """Finite descriptive statistics with sample standard deviation."""

    count: int = Field(gt=0)
    mean: float
    standard_deviation: float = Field(ge=0.0)
    minimum: float
    maximum: float


class Day6SeedResult(StrictModel):
    """One unchanged-configuration training result, including failures."""

    root_seed: int = Field(ge=0)
    status: Literal["complete", "failed"]
    metrics: dict[str, float] | None = None
    failure: str | None = None

    @model_validator(mode="after")
    def result_matches_status(self) -> Self:
        if self.status == "complete" and (self.metrics is None or self.failure is not None):
            raise ValueError("complete seed results require metrics and cannot contain a failure")
        if self.status == "failed" and (not self.failure or self.metrics is not None):
            raise ValueError("failed seed results require a failure and cannot contain metrics")
        return self


class AlignedPolicyEvaluation(StrictModel):
    """Per-episode policy values whose root and evaluation seeds remain explicit."""

    root_seed: int = Field(ge=0)
    policy: str = Field(min_length=1)
    seed_partition: Literal["train", "validation", "test"]
    evaluation_seeds: tuple[int, ...] = Field(min_length=1)
    metrics: dict[str, tuple[float, ...]]

    @model_validator(mode="after")
    def metric_lengths_match_seeds(self) -> Self:
        count = len(self.evaluation_seeds)
        if len(set(self.evaluation_seeds)) != count:
            raise ValueError("evaluation seeds must be unique")
        invalid = [name for name, values in self.metrics.items() if len(values) != count]
        if invalid:
            raise ValueError(f"metric values do not align with evaluation seeds: {invalid}")
        return self


class PairedMetricDifference(StrictModel):
    """Seed-aligned left-minus-right metric values and their summary."""

    root_seed: int = Field(ge=0)
    metric: str
    left_policy: str
    right_policy: str
    evaluation_seeds: tuple[int, ...]
    values: tuple[float, ...]
    summary: MetricStatistics


class CandidateEvidence(StrictModel):
    """Aggregate evidence offered for validation-only candidate selection."""

    name: str = Field(min_length=1)
    source_configuration: str = Field(min_length=1)
    seed_partition: Literal["train", "validation", "test"]
    training_seeds: tuple[int, ...]
    metrics: dict[str, float]


class CandidateFreezeRecord(StrictModel):
    """Committed identity and evidence boundary for the frozen candidate."""

    schema_version: int = Field(default=1, ge=1, le=1)
    source_configuration: str
    selection_metric: str
    selection_reason: str
    alternatives_evaluated: tuple[str, ...]
    configuration_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    reward_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    environment_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    git_commit: str = Field(min_length=7)


def summarize_values(values: Sequence[float]) -> MetricStatistics:
    """Summarize every supplied value without bootstrapping or dropping failures."""

    if not values:
        raise TrainingError("cannot summarize an empty metric")
    finite_values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in finite_values):
        raise TrainingError("cannot summarize NaN or infinite metric values")
    return MetricStatistics(
        count=len(finite_values),
        mean=statistics.fmean(finite_values),
        standard_deviation=(statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0),
        minimum=min(finite_values),
        maximum=max(finite_values),
    )


def aggregate_training_seed_metric(
    results: Sequence[Day6SeedResult],
    metric: str,
    *,
    expected_root_seeds: tuple[int, ...] = DAY6_ROOT_SEEDS,
) -> MetricStatistics:
    """Aggregate exactly the declared seeds and reject missing or failed runs."""

    by_seed = _results_by_seed(results, expected_root_seeds)
    failed = [seed for seed, result in by_seed.items() if result.status == "failed"]
    if failed:
        raise TrainingError(f"cannot aggregate failed Day 6 seeds: {failed}")
    missing_metric = [
        seed
        for seed, result in by_seed.items()
        if result.metrics is None or metric not in result.metrics
    ]
    if missing_metric:
        raise TrainingError(f"Day 6 seed results are missing metric {metric}: {missing_metric}")
    return summarize_values(
        [
            by_seed[seed].metrics[metric]  # type: ignore[index]
            for seed in expected_root_seeds
        ]
    )


def paired_metric_difference(
    left: AlignedPolicyEvaluation,
    right: AlignedPolicyEvaluation,
    metric: str,
) -> PairedMetricDifference:
    """Subtract policies only after checking root, partition, and seed alignment."""

    if left.root_seed != right.root_seed:
        raise TrainingError("paired policy evaluations have different root seeds")
    if left.seed_partition != right.seed_partition:
        raise TrainingError("paired policy evaluations use different seed partitions")
    if left.evaluation_seeds != right.evaluation_seeds:
        raise TrainingError("paired policy evaluations have different evaluation seeds")
    if metric not in left.metrics or metric not in right.metrics:
        raise TrainingError(f"paired policy evaluations are missing metric {metric}")
    values = tuple(
        left_value - right_value
        for left_value, right_value in zip(
            left.metrics[metric],
            right.metrics[metric],
            strict=True,
        )
    )
    return PairedMetricDifference(
        root_seed=left.root_seed,
        metric=metric,
        left_policy=left.policy,
        right_policy=right.policy,
        evaluation_seeds=left.evaluation_seeds,
        values=values,
        summary=summarize_values(values),
    )


def load_day6_training_inputs(path: Path, *, root_seed: int) -> TrainingInputs:
    """Load one Day 6 seed while categorically excluding final-test evaluation."""

    if root_seed not in DAY6_ROOT_SEEDS:
        raise ConfigurationError(f"Day 6 root seed must be one of {DAY6_ROOT_SEEDS}")
    inputs = load_training_inputs(path)
    require_validation_partition(inputs.validation)
    training = inputs.training.model_copy(update={"seed": root_seed})
    return TrainingInputs(training, inputs.environment, inputs.validation)


def require_validation_partition(evaluation: EvaluationConfig) -> None:
    """Reject train or final-test evidence in every Day 6 selection workflow."""

    if evaluation.seed_partition != "validation":
        raise ConfigurationError(
            "Day 6 commands require the validation partition; final-test seeds are forbidden"
        )


def training_configuration_differences(
    left: TrainingConfig,
    right: TrainingConfig,
) -> tuple[str, ...]:
    """Return deterministic dotted paths for every changed training field."""

    left_data = left.model_dump(mode="json", exclude_none=False)
    right_data = right.model_dump(mode="json", exclude_none=False)
    return tuple(_mapping_differences(left_data, right_data))


def reward_fingerprint(environment: EnvConfig) -> str:
    """Hash the validated reward definition independently from other environment fields."""

    return configuration_fingerprint(environment.reward)


def select_validation_candidate(candidates: Sequence[CandidateEvidence]) -> CandidateEvidence:
    """Select using aggregate validation metrics and all three controlled seeds only."""

    if not candidates:
        raise TrainingError("candidate selection requires at least one alternative")
    for candidate in candidates:
        if candidate.seed_partition != "validation":
            raise TrainingError("candidate selection may use validation metrics only")
        if candidate.training_seeds != DAY6_ROOT_SEEDS:
            raise TrainingError(
                f"candidate evidence must include exact training seeds {DAY6_ROOT_SEEDS}"
            )
        _candidate_selection_key(candidate.metrics)
    return max(candidates, key=lambda candidate: _candidate_selection_key(candidate.metrics))


def candidate_configuration_fingerprint(path: Path) -> str:
    """Fingerprint the validated candidate, including every mutable research field."""

    return training_definition_fingerprint(load_training_inputs(path).training)


def validate_candidate_freeze(
    candidate_path: Path,
    freeze: CandidateFreezeRecord,
) -> None:
    """Detect candidate, reward, environment, seed, or partition mutation."""

    inputs = load_training_inputs(candidate_path)
    require_validation_partition(inputs.validation)
    actual = (
        training_definition_fingerprint(inputs.training),
        reward_fingerprint(inputs.environment),
        configuration_fingerprint(inputs.environment),
        inputs.validation.resolved_seeds,
    )
    expected = (
        freeze.configuration_fingerprint,
        freeze.reward_fingerprint,
        freeze.environment_fingerprint,
        freeze.validation_seeds,
    )
    if actual != expected:
        raise ConfigurationError("frozen candidate identity does not match its freeze record")
    if freeze.training_seeds != DAY6_ROOT_SEEDS:
        raise ConfigurationError("frozen candidate does not declare all Day 6 training seeds")


def load_candidate_freeze(path: Path) -> CandidateFreezeRecord:
    """Load a strict candidate freeze record with contextual errors."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CandidateFreezeRecord.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ConfigurationError(f"cannot load candidate freeze record {path}: {exc}") from exc


def _results_by_seed(
    results: Sequence[Day6SeedResult],
    expected_root_seeds: tuple[int, ...],
) -> dict[int, Day6SeedResult]:
    by_seed = {result.root_seed: result for result in results}
    identities = tuple(sorted(by_seed))
    expected = tuple(sorted(expected_root_seeds))
    if len(by_seed) != len(results):
        raise TrainingError("Day 6 seed results contain duplicate root-seed identities")
    if identities != expected:
        raise TrainingError(
            f"Day 6 seed identities do not match: expected={expected}, actual={identities}"
        )
    return by_seed


def _mapping_differences(
    left: Mapping[str, object],
    right: Mapping[str, object],
    prefix: str = "",
) -> list[str]:
    differences: list[str] = []
    for key in sorted(set(left) | set(right)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in left or key not in right:
            differences.append(path)
            continue
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            differences.extend(_mapping_differences(left_value, right_value, path))
        elif left_value != right_value:
            differences.append(path)
    return differences


def _candidate_selection_key(metrics: Mapping[str, float]) -> tuple[float, ...]:
    required = (
        "success_rate",
        "targets_recovered",
        "team_efficiency",
        "exploration_coverage",
        "duplicated_exploration",
        "episode_length",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise TrainingError(f"candidate evidence is missing validation metrics: {missing}")
    return (
        metrics["success_rate"],
        metrics["targets_recovered"],
        metrics["team_efficiency"],
        metrics["exploration_coverage"],
        -metrics["duplicated_exploration"],
        -metrics["episode_length"],
    )
