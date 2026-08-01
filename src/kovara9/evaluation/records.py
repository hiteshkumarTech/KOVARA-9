"""Typed evaluation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Metrics from one explicitly seeded episode."""

    seed: int
    success: bool
    episode_length: int
    targets_recovered: int
    total_targets: int
    exploration_coverage: float
    duplicated_exploration: float
    communication_messages: int
    communication_rejections: int
    messages_per_agent_step: float
    team_efficiency: float
    shared_return: float
    termination_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive statistics for one numeric metric."""

    count: int
    mean: float
    standard_deviation: float
    median: float
    confidence_low: float | None
    confidence_high: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate suite report."""

    name: str
    policy: str
    episode_count: int
    metrics: dict[str, MetricSummary]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""

        return {
            "name": self.name,
            "policy": self.policy,
            "episode_count": self.episode_count,
            "metrics": {name: summary.to_dict() for name, summary in sorted(self.metrics.items())},
        }
