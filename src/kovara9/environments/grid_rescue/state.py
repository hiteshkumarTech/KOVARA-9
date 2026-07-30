"""Mutable simulator state kept behind immutable snapshot boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from kovara9.core.types import AgentId, BoolArray, Position, WorldSnapshot


@dataclass(slots=True)
class WorldState:
    """Internal state owned exclusively by one environment."""

    width: int
    height: int
    obstacles: BoolArray
    agent_positions: dict[AgentId, Position]
    targets: set[Position]
    recovered_targets: set[Position]
    communication_budgets: dict[AgentId, int]
    latest_messages: dict[AgentId, int]
    step_count: int = 0

    def snapshot(self) -> WorldSnapshot:
        """Return defensive, read-only copies for external consumers."""

        obstacles = self.obstacles.copy()
        obstacles.flags.writeable = False
        return WorldSnapshot(
            width=self.width,
            height=self.height,
            obstacles=obstacles,
            agent_positions=dict(self.agent_positions),
            targets=frozenset(self.targets),
            recovered_targets=frozenset(self.recovered_targets),
            communication_budgets=dict(self.communication_budgets),
            latest_messages=dict(self.latest_messages),
            step_count=self.step_count,
        )
