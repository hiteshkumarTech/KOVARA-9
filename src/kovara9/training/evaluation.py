"""Decentralized deterministic policy adapters for learned actor evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from gymnasium import spaces
from gymnasium.spaces import Space

from kovara9.agents.policy import Policy, PolicyTransitionInfo
from kovara9.core.errors import TrainingError
from kovara9.core.types import AgentId
from kovara9.training.encoding import ActorObservationEncoder
from kovara9.training.networks import SharedActor
from kovara9.training.policy import select_actions


class DecentralizedActorPolicy:
    """Evaluation-only policy that cannot accept centralized state."""

    def __init__(
        self,
        *,
        actor: SharedActor,
        device: torch.device,
        policy_name: str,
        parameters: Mapping[str, bool | float | int | str],
    ) -> None:
        self.actor = actor.to(device)
        self.device = device
        self._name = policy_name
        self._parameters = MappingProxyType(dict(parameters))
        self._encoder: ActorObservationEncoder | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Mapping[str, bool | float | int | str]:
        return self._parameters

    def reset(
        self,
        *,
        agent_id: AgentId,
        observation_space: Space[Any],
        action_space: Space[Any],
        seed: int,
    ) -> None:
        """Validate the local contract; deterministic inference consumes no RNG."""

        if not agent_id:
            raise TrainingError("checkpoint policy requires a non-empty agent identifier")
        if seed < 0:
            raise TrainingError("checkpoint policy reset seed must be non-negative")
        if not isinstance(action_space, spaces.Dict):
            raise TrainingError("checkpoint policy action space must be a Dict")
        move_space = action_space.spaces.get("move")
        message_space = action_space.spaces.get("message")
        if not isinstance(move_space, spaces.Discrete) or not isinstance(
            message_space, spaces.Discrete
        ):
            raise TrainingError("checkpoint policy requires discrete move and message actions")
        encoder = ActorObservationEncoder(observation_space)
        if encoder.input_dim != self.actor.input_dim:
            raise TrainingError("checkpoint actor input does not match observation signature")
        if encoder.move_action_count != self.actor.move_action_count:
            raise TrainingError("checkpoint movement head does not match action signature")
        if encoder.message_action_count != self.actor.message_action_count:
            raise TrainingError("checkpoint message head does not match action signature")
        if move_space.n != self.actor.move_action_count:
            raise TrainingError("checkpoint movement action space does not match actor head")
        if message_space.n != self.actor.message_action_count:
            raise TrainingError("checkpoint message action space does not match actor head")
        self._encoder = encoder

    def act(self, observation: dict[str, Any]) -> dict[str, int]:
        """Choose masked modes from one local observation only."""

        if self._encoder is None:
            raise TrainingError("checkpoint policy must be reset before acting")
        encoded = self._encoder.encode([observation], device=self.device)
        self.actor.eval()
        with torch.no_grad():
            statistics = select_actions(
                self.actor,
                encoded,
                deterministic=True,
            ).statistics
        return {
            "move": int(statistics.move_actions.item()),
            "message": int(statistics.message_actions.item()),
        }

    def observe_outcome(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: PolicyTransitionInfo,
    ) -> None:
        """Remain feed-forward and stateless across transitions."""


def actor_policy_factory(
    *,
    actor: SharedActor,
    device: torch.device,
    policy_name: str,
    parameters: Mapping[str, bool | float | int | str],
) -> Callable[[], Policy]:
    """Create independent stateless policy shells sharing one frozen actor."""

    def factory() -> Policy:
        return DecentralizedActorPolicy(
            actor=actor,
            device=device,
            policy_name=policy_name,
            parameters=parameters,
        )

    return factory
