"""Adapters from declared observation/state spaces to learner tensor contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.spaces import Space
from torch import Tensor

from kovara9.core.errors import TrainingError
from kovara9.training.networks import ActorInput, CriticInput


@dataclass(frozen=True, slots=True)
class EncodedActorBatch:
    """Actor features and independently factored legality masks."""

    inputs: ActorInput
    move_action_masks: Tensor
    message_action_masks: Tensor


def _require_dict_space(name: str, space: Space[Any]) -> spaces.Dict:
    if not isinstance(space, spaces.Dict):
        raise TypeError(f"{name} must be a gymnasium Dict space")
    return space


def _require_space[SpaceT: Space[Any]](
    parent: spaces.Dict,
    key: str,
    expected: type[SpaceT],
) -> SpaceT:
    child = parent[key]
    if not isinstance(child, expected):
        raise TypeError(f"{key} must use {expected.__name__}")
    return child


def _one_hot(values: np.ndarray[Any, np.dtype[np.int64]], counts: tuple[int, ...]) -> np.ndarray:
    encoded = [np.eye(count, dtype=np.float32)[values[index]] for index, count in enumerate(counts)]
    return np.concatenate(encoded, dtype=np.float32)


class ActorObservationEncoder:
    """Encode only the documented decentralized observation fields."""

    def __init__(self, observation_space: Space[Any]) -> None:
        self.observation_space = _require_dict_space("observation_space", observation_space)
        self.local_grid = _require_space(self.observation_space, "local_grid", spaces.Box)
        self.active_agents = _require_space(
            self.observation_space, "active_agents", spaces.MultiBinary
        )
        self.messages = _require_space(self.observation_space, "messages", spaces.MultiDiscrete)
        self.communication_budget = _require_space(
            self.observation_space, "communication_budget", spaces.Discrete
        )
        self.move_mask = _require_space(
            self.observation_space, "move_action_mask", spaces.MultiBinary
        )
        self.message_mask = _require_space(
            self.observation_space, "message_action_mask", spaces.MultiBinary
        )
        self.message_counts = tuple(int(count) for count in self.messages.nvec.reshape(-1))
        self.input_dim = (
            int(np.prod(self.local_grid.shape))
            + int(np.prod(self.active_agents.shape))
            + sum(self.message_counts)
            + 1
        )
        self.move_action_count = int(np.prod(self.move_mask.shape))
        self.message_action_count = int(np.prod(self.message_mask.shape))

    def encode(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> EncodedActorBatch:
        """Validate and encode a batch without accepting centralized fields."""

        if not observations:
            raise TrainingError("actor observation batch cannot be empty")
        features: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        move_masks: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
        message_masks: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
        budget_scale = max(int(self.communication_budget.n) - 1, 1)
        for index, observation in enumerate(observations):
            candidate = dict(observation)
            if not self.observation_space.contains(candidate):
                raise TrainingError(
                    f"actor observation {index} does not match the decentralized observation space"
                )
            local_grid = np.asarray(candidate["local_grid"], dtype=np.float32).reshape(-1)
            active_agents = np.asarray(candidate["active_agents"], dtype=np.float32).reshape(-1)
            messages = np.asarray(candidate["messages"], dtype=np.int64).reshape(-1)
            message_features = _one_hot(messages, self.message_counts)
            budget = np.asarray(
                [float(candidate["communication_budget"]) / budget_scale],
                dtype=np.float32,
            )
            features.append(
                np.concatenate(
                    (local_grid, active_agents, message_features, budget),
                    dtype=np.float32,
                )
            )
            move_masks.append(np.asarray(candidate["move_action_mask"], dtype=np.bool_).reshape(-1))
            message_masks.append(
                np.asarray(candidate["message_action_mask"], dtype=np.bool_).reshape(-1)
            )
        return EncodedActorBatch(
            inputs=ActorInput(
                torch.as_tensor(np.stack(features), dtype=torch.float32, device=device)
            ),
            move_action_masks=torch.as_tensor(
                np.stack(move_masks), dtype=torch.bool, device=device
            ),
            message_action_masks=torch.as_tensor(
                np.stack(message_masks), dtype=torch.bool, device=device
            ),
        )


class CentralStateEncoder:
    """Encode the trainer-only centralized state contract for the critic."""

    def __init__(self, state_space: Space[Any]) -> None:
        self.state_space = _require_dict_space("state_space", state_space)
        self.global_grid = _require_space(self.state_space, "global_grid", spaces.Box)
        self.active_agents = _require_space(self.state_space, "active_agents", spaces.MultiBinary)
        self.communication_budgets = _require_space(
            self.state_space, "communication_budgets", spaces.Box
        )
        self.latest_messages = _require_space(
            self.state_space, "latest_messages", spaces.MultiDiscrete
        )
        self.step_count = _require_space(self.state_space, "step_count", spaces.Discrete)
        self.message_counts = tuple(int(count) for count in self.latest_messages.nvec.reshape(-1))
        self.input_dim = (
            int(np.prod(self.global_grid.shape))
            + int(np.prod(self.active_agents.shape))
            + int(np.prod(self.communication_budgets.shape))
            + sum(self.message_counts)
            + 1
        )

    def encode(
        self,
        states: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> CriticInput:
        """Validate and encode a batch of centralized states."""

        if not states:
            raise TrainingError("centralized state batch cannot be empty")
        features: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        budget_high = np.maximum(
            np.asarray(self.communication_budgets.high, dtype=np.float32).reshape(-1),
            1.0,
        )
        step_scale = max(int(self.step_count.n) - 1, 1)
        for index, state in enumerate(states):
            candidate = dict(state)
            if not self.state_space.contains(candidate):
                raise TrainingError(
                    f"centralized state {index} does not match the critic state space"
                )
            global_grid = np.asarray(candidate["global_grid"], dtype=np.float32).reshape(-1)
            active_agents = np.asarray(candidate["active_agents"], dtype=np.float32).reshape(-1)
            budgets = (
                np.asarray(candidate["communication_budgets"], dtype=np.float32).reshape(-1)
                / budget_high
            )
            messages = np.asarray(candidate["latest_messages"], dtype=np.int64).reshape(-1)
            message_features = _one_hot(messages, self.message_counts)
            step = np.asarray([float(candidate["step_count"]) / step_scale], dtype=np.float32)
            features.append(
                np.concatenate(
                    (global_grid, active_agents, budgets, message_features, step),
                    dtype=np.float32,
                )
            )
        return CriticInput(torch.as_tensor(np.stack(features), dtype=torch.float32, device=device))
