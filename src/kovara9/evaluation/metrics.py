"""Metric formulas and deterministic aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from kovara9.config.models import EvaluationConfig
from kovara9.core.seeding import derive_seed
from kovara9.evaluation.records import EpisodeRecord, EvaluationSummary, MetricSummary


def exploration_coverage[CellT](
    observed_by_agent: Iterable[set[CellT]],
    reachable_cell_count: int,
) -> float:
    """Fraction of reachable cells observed by at least one agent."""

    if reachable_cell_count <= 0:
        raise ValueError("reachable_cell_count must be positive")
    union: set[CellT] = set()
    for observed in observed_by_agent:
        union.update(observed)
    return len(union) / reachable_cell_count


def duplicated_exploration[CellT](observed_by_agent: Iterable[set[CellT]]) -> float:
    """Fraction of per-agent unique observations redundant with teammates."""

    observations = list(observed_by_agent)
    total = sum(len(observed) for observed in observations)
    if total == 0:
        return 0.0
    union: set[CellT] = set()
    for observed in observations:
        union.update(observed)
    return (total - len(union)) / total


def team_efficiency(targets_recovered: int, agent_steps: int) -> float:
    """Targets recovered per active agent-step."""

    if agent_steps <= 0:
        return 0.0
    return targets_recovered / agent_steps


def generalization_gap(
    reference_records: Sequence[EpisodeRecord],
    held_out_records: Sequence[EpisodeRecord],
) -> float:
    """Reference success rate minus held-out success rate."""

    if not reference_records or not held_out_records:
        raise ValueError("both record suites must be non-empty")
    reference = np.mean([record.success for record in reference_records])
    held_out = np.mean([record.success for record in held_out_records])
    return float(reference - held_out)


def _summarize_values(
    values: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> MetricSummary:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty metric")
    low: float | None = None
    high: float | None = None
    if samples > 0:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, array.size, size=(samples, array.size))
        means = np.mean(array[indices], axis=1)
        alpha = (1.0 - confidence) / 2.0
        low, high = (float(value) for value in np.quantile(means, [alpha, 1.0 - alpha]))
    return MetricSummary(
        count=int(array.size),
        mean=float(np.mean(array)),
        standard_deviation=float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        median=float(np.median(array)),
        confidence_low=low,
        confidence_high=high,
    )


def aggregate_records(
    records: Sequence[EpisodeRecord],
    config: EvaluationConfig,
    policy_name: str,
) -> EvaluationSummary:
    """Aggregate all documented metrics without inferring missing results."""

    if not records:
        raise ValueError("evaluation produced no episode records")
    metric_values = {
        "success_rate": [float(record.success) for record in records],
        "episode_length": [float(record.episode_length) for record in records],
        "exploration_coverage": [record.exploration_coverage for record in records],
        "duplicated_exploration": [record.duplicated_exploration for record in records],
        "communication_messages": [float(record.communication_messages) for record in records],
        "messages_per_agent_step": [record.messages_per_agent_step for record in records],
        "team_efficiency": [record.team_efficiency for record in records],
        "shared_return": [record.shared_return for record in records],
    }
    metrics = {
        name: _summarize_values(
            values,
            samples=config.bootstrap_samples,
            confidence=config.bootstrap_confidence,
            seed=derive_seed(0, config.name, policy_name, name, "bootstrap"),
        )
        for name, values in metric_values.items()
    }
    return EvaluationSummary(
        name=config.name,
        policy=policy_name,
        episode_count=len(records),
        metrics=metrics,
    )
