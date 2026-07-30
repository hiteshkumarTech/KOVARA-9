"""Seeded random baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Discrete, Space

from kovara9.core.types import AgentId


class RandomPolicy:
    """Uniform movement with sparse valid random communication."""

    def __init__(self, message_probability: float = 0.1) -> None:
        if not 0.0 <= message_probability <= 1.0:
            raise ValueError("message_probability must be in [0, 1]")
        self.message_probability = message_probability
        self._rng = np.random.default_rng(0)
        self._move_count = 0
        self._message_count = 1

    @property
    def name(self) -> str:
        return "random"

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
            raise TypeError("RandomPolicy requires a dictionary action space")
        move_space = action_space["move"]
        message_space = action_space["message"]
        if not isinstance(move_space, Discrete) or not isinstance(message_space, Discrete):
            raise TypeError("RandomPolicy requires discrete move and message spaces")
        self._move_count = int(move_space.n)
        self._message_count = int(message_space.n)
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, Any]) -> dict[str, int]:
        budget = int(observation["communication_budget"])
        should_send = (
            self._message_count > 1 and budget > 0 and self._rng.random() < self.message_probability
        )
        message = int(self._rng.integers(1, self._message_count)) if should_send else 0
        return {
            "move": int(self._rng.integers(0, self._move_count)),
            "message": message,
        }

    def observe_outcome(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        del reward, terminated, truncated, info
