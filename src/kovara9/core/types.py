"""Typed values shared across simulator subsystems."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

type AgentId = str
type BoolArray = NDArray[np.bool_]
type UInt8Array = NDArray[np.uint8]

MAX_AGENTS = 4


@dataclass(frozen=True, order=True, slots=True)
class Position:
    """A row/column location in the grid."""

    row: int
    col: int

    def moved(self, row_delta: int, col_delta: int) -> Position:
        """Return a translated position."""

        return Position(self.row + row_delta, self.col + col_delta)


class Move(IntEnum):
    """Discrete movement commands."""

    STAY = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4

    @property
    def delta(self) -> tuple[int, int]:
        """Return the row/column delta for this move."""

        return {
            Move.STAY: (0, 0),
            Move.NORTH: (-1, 0),
            Move.EAST: (0, 1),
            Move.SOUTH: (1, 0),
            Move.WEST: (0, -1),
        }[self]


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One agent's simultaneous movement and communication action."""

    move: Move
    message: int = 0


@dataclass(frozen=True, slots=True)
class StepEvents:
    """Facts emitted by a completed simulator transition."""

    recovered_targets: tuple[Position, ...]
    messages_sent: int
    blocked_agents: tuple[AgentId, ...]


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """Immutable copy of world state for rendering and reporting."""

    width: int
    height: int
    obstacles: BoolArray
    agent_positions: dict[AgentId, Position]
    targets: frozenset[Position]
    recovered_targets: frozenset[Position]
    communication_budgets: dict[AgentId, int]
    latest_messages: dict[AgentId, int]
    step_count: int
