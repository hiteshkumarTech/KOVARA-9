"""Algorithm-independent future CTDE trainer protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from kovara9.agents.policy import Policy


class Trainer(Protocol):
    """Boundary future learning algorithms must implement."""

    @property
    def name(self) -> str:
        """Stable algorithm name."""

        ...

    def train(self, *, output_dir: Path, seed: int) -> tuple[Policy, ...]:
        """Train and return independently executable policies."""

        ...
