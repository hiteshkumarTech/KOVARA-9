"""Algorithm-independent future CTDE trainer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    """Serializable counters required for checkpoint resume."""

    environment_steps: int
    optimizer_updates: int
    completed_episodes: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Result of a completed or deliberately bounded training invocation."""

    checkpoint: Path
    progress: TrainingProgress


class Trainer(Protocol):
    """Boundary future learning algorithms must implement."""

    @property
    def name(self) -> str:
        """Stable algorithm name."""

        ...

    def train(
        self,
        *,
        output_dir: Path,
        resume_from: Path | None = None,
    ) -> TrainingResult:
        """Train one shared policy and return its checkpoint and exact progress."""

        ...
