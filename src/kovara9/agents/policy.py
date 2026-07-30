"""Decentralized policy protocol."""

from __future__ import annotations

from typing import Any, Protocol

from gymnasium.spaces import Space

from kovara9.core.types import AgentId


class Policy(Protocol):
    """One independently executing, optionally stateful agent policy."""

    @property
    def name(self) -> str:
        """Stable policy name for artifact provenance."""

        ...

    def reset(
        self,
        *,
        agent_id: AgentId,
        observation_space: Space[Any],
        action_space: Space[Any],
        seed: int,
    ) -> None:
        """Reset all private state for a new episode."""

        ...

    def act(self, observation: dict[str, Any]) -> dict[str, int]:
        """Select an action from only this agent's observation and memory."""

        ...

    def observe_outcome(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Receive this agent's public transition result."""

        ...
