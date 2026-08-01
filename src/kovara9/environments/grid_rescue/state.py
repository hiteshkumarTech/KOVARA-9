"""Mutable simulator state kept behind immutable snapshot boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from pydantic import Field, model_validator

from kovara9.config.models import StrictModel
from kovara9.core.types import AgentId, BoolArray, Position, WorldSnapshot


class CheckpointPosition(StrictModel):
    """JSON-safe position used at the simulator checkpoint boundary."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)

    @classmethod
    def from_position(cls, position: Position) -> CheckpointPosition:
        """Copy a runtime position into its validated persistence form."""

        return cls(row=position.row, col=position.col)

    def to_position(self) -> Position:
        """Convert the validated persistence form back to a runtime position."""

        return Position(self.row, self.col)


class WorldStateCheckpoint(StrictModel):
    """Validated, implementation-neutral snapshot of mutable simulator state."""

    schema_version: int = Field(default=1, ge=1, le=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    obstacles: tuple[tuple[bool, ...], ...]
    agent_positions: dict[AgentId, CheckpointPosition]
    targets: tuple[CheckpointPosition, ...]
    recovered_targets: tuple[CheckpointPosition, ...]
    communication_budgets: dict[AgentId, int]
    latest_messages: dict[AgentId, int]
    step_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_shape_and_membership(self) -> Self:
        """Reject malformed state before it can enter the simulator."""

        if len(self.obstacles) != self.height or any(
            len(row) != self.width for row in self.obstacles
        ):
            raise ValueError("checkpoint obstacle grid does not match width and height")
        agent_ids = set(self.agent_positions)
        if set(self.communication_budgets) != agent_ids:
            raise ValueError("checkpoint communication budgets do not match agent positions")
        if set(self.latest_messages) != agent_ids:
            raise ValueError("checkpoint latest messages do not match agent positions")
        targets = set(self.targets)
        if not set(self.recovered_targets).issubset(targets):
            raise ValueError("checkpoint recovered targets are not a subset of targets")
        for position in (*self.agent_positions.values(), *self.targets):
            if position.row >= self.height or position.col >= self.width:
                raise ValueError("checkpoint position is outside the world bounds")
            if self.obstacles[position.row][position.col]:
                raise ValueError("checkpoint places an agent or target on an obstacle")
        if any(budget < 0 for budget in self.communication_budgets.values()):
            raise ValueError("checkpoint communication budgets cannot be negative")
        if any(message < 0 for message in self.latest_messages.values()):
            raise ValueError("checkpoint messages cannot be negative")
        return self


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

    def checkpoint(self) -> WorldStateCheckpoint:
        """Export state without exposing the mutable simulator object itself."""

        return WorldStateCheckpoint(
            width=self.width,
            height=self.height,
            obstacles=tuple(tuple(bool(cell) for cell in row) for row in self.obstacles),
            agent_positions={
                agent: CheckpointPosition.from_position(position)
                for agent, position in self.agent_positions.items()
            },
            targets=tuple(
                CheckpointPosition.from_position(position) for position in sorted(self.targets)
            ),
            recovered_targets=tuple(
                CheckpointPosition.from_position(position)
                for position in sorted(self.recovered_targets)
            ),
            communication_budgets=dict(self.communication_budgets),
            latest_messages=dict(self.latest_messages),
            step_count=self.step_count,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: WorldStateCheckpoint) -> WorldState:
        """Build fresh mutable state from a validated checkpoint snapshot."""

        return cls(
            width=checkpoint.width,
            height=checkpoint.height,
            obstacles=np.asarray(checkpoint.obstacles, dtype=np.bool_),
            agent_positions={
                agent: position.to_position()
                for agent, position in checkpoint.agent_positions.items()
            },
            targets={position.to_position() for position in checkpoint.targets},
            recovered_targets={position.to_position() for position in checkpoint.recovered_targets},
            communication_budgets=dict(checkpoint.communication_budgets),
            latest_messages=dict(checkpoint.latest_messages),
            step_count=checkpoint.step_count,
        )
