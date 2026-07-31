"""Decentralized policy protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

from gymnasium.spaces import Space

from kovara9.core.types import AgentId


class PolicyTransitionInfo(TypedDict):
    """Personal transition facts permitted at the decentralized policy boundary."""

    blocked: bool
    message_sent: bool
    communication_rejected: bool


class Policy(Protocol):
    """One independently executing, optionally stateful agent policy."""

    @property
    def name(self) -> str:
        """Stable policy name for artifact provenance."""

        ...

    @property
    def parameters(self) -> Mapping[str, bool | float | int | str]:
        """Validated policy parameters that affect behavior."""

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
        info: PolicyTransitionInfo,
    ) -> None:
        """Receive this agent's public transition result."""

        ...
