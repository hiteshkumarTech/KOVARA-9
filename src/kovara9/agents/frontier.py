"""Deterministic decentralized exploration heuristic."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Discrete, Space

from kovara9.agents.policy import PolicyTransitionInfo
from kovara9.core.types import AgentId, Move, Position


class FrontierPolicy:
    """Prefer visible targets, then least-visited feasible directions."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._position = Position(0, 0)
        self._visits: dict[Position, int] = {}
        self._last_move = Move.STAY
        self._message_count = 1

    @property
    def name(self) -> str:
        return "frontier"

    @property
    def parameters(self) -> dict[str, bool | float | int | str]:
        return {}

    def reset(
        self,
        *,
        agent_id: AgentId,
        observation_space: Space[Any],
        action_space: Space[Any],
        seed: int,
    ) -> None:
        del agent_id, observation_space
        if not isinstance(action_space, DictSpace):
            raise TypeError("FrontierPolicy requires a dictionary action space")
        message_space = action_space["message"]
        if not isinstance(message_space, Discrete):
            raise TypeError("FrontierPolicy requires a discrete message space")
        self._rng = np.random.default_rng(seed)
        self._position = Position(0, 0)
        self._visits = {self._position: 1}
        self._last_move = Move.STAY
        self._message_count = int(message_space.n)

    def act(self, observation: dict[str, Any]) -> dict[str, int]:
        grid = np.asarray(observation["local_grid"], dtype=np.uint8)
        obstacle = grid[1]
        targets = grid[2]
        center = obstacle.shape[0] // 2
        move = self._move_toward_visible_target(obstacle, targets, center)
        if move is None:
            move = self._least_visited_move(obstacle, center)
        self._last_move = move

        target_visible = bool(np.any(targets))
        can_send = int(observation["communication_budget"]) > 0 and self._message_count > 1
        message = 1 if target_visible and can_send else 0
        return {"move": int(move), "message": message}

    def observe_outcome(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: PolicyTransitionInfo,
    ) -> None:
        del reward, terminated, truncated
        if not bool(info.get("blocked", False)):
            row_delta, col_delta = self._last_move.delta
            self._position = self._position.moved(row_delta, col_delta)
        self._visits[self._position] = self._visits.get(self._position, 0) + 1

    @staticmethod
    def _move_toward_visible_target(
        obstacles: np.ndarray[Any, np.dtype[np.uint8]],
        targets: np.ndarray[Any, np.dtype[np.uint8]],
        center: int,
    ) -> Move | None:
        start = (center, center)
        queue = deque([(start, Move.STAY)])
        visited = {start}
        deltas = (
            (Move.NORTH, -1, 0),
            (Move.EAST, 0, 1),
            (Move.SOUTH, 1, 0),
            (Move.WEST, 0, -1),
        )
        while queue:
            (row, col), first_move = queue.popleft()
            if targets[row, col] and (row, col) != start:
                return first_move
            for move, row_delta, col_delta in deltas:
                neighbor = (row + row_delta, col + col_delta)
                if (
                    0 <= neighbor[0] < obstacles.shape[0]
                    and 0 <= neighbor[1] < obstacles.shape[1]
                    and not obstacles[neighbor]
                    and neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append((neighbor, move if first_move is Move.STAY else first_move))
        return None

    def _least_visited_move(
        self,
        obstacles: np.ndarray[Any, np.dtype[np.uint8]],
        center: int,
    ) -> Move:
        candidates: list[tuple[int, Move]] = []
        for move in (Move.NORTH, Move.EAST, Move.SOUTH, Move.WEST):
            row_delta, col_delta = move.delta
            if not obstacles[center + row_delta, center + col_delta]:
                destination = self._position.moved(row_delta, col_delta)
                candidates.append((self._visits.get(destination, 0), move))
        if not candidates:
            return Move.STAY
        minimum = min(visits for visits, _move in candidates)
        best = [move for visits, move in candidates if visits == minimum]
        return best[int(self._rng.integers(0, len(best)))]
