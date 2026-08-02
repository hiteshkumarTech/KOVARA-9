"""Synchronous in-process rollout collection across declared environment adapters."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from gymnasium.spaces import Space
from torch import Tensor

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.core.types import AgentId, StepEvents
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.policy import select_actions
from kovara9.training.rollout import RolloutBatch, RolloutBuffer, RolloutSpec, RolloutStep
from kovara9.training.runtime import make_torch_generator
from kovara9.training.seeds import ExperimentSeedStreams

type Observation = dict[str, Any]
type ObservationBatch = dict[AgentId, Observation]
type InfoBatch = dict[AgentId, dict[str, Any]]
type ActionBatch = dict[AgentId, dict[str, int]]


class RolloutEnvironment(Protocol):
    """Minimal parallel-environment surface required by the collector."""

    possible_agents: list[AgentId]
    agents: list[AgentId]
    state_space: Space[Any]
    last_events: StepEvents

    def observation_space(self, agent: AgentId) -> Space[Any]: ...

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationBatch, InfoBatch]: ...

    def step(
        self,
        actions: Mapping[AgentId, Mapping[str, Any]],
    ) -> tuple[
        ObservationBatch,
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        InfoBatch,
    ]: ...

    def state(self) -> dict[str, Any]: ...

    def checkpoint_state(self) -> dict[str, Any]: ...

    def restore_checkpoint_state(
        self,
        raw_checkpoint: Mapping[str, Any],
    ) -> ObservationBatch: ...

    def close(self) -> None: ...


type EnvironmentFactory = Callable[[], RolloutEnvironment]


@dataclass(frozen=True, slots=True)
class CompletedEpisode:
    """One boundary observed while collecting a fixed-length rollout."""

    environment_id: int
    episode_index: int
    reset_seed: int
    length: int
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class RolloutCollection:
    """Tensor rollout plus the exact episode/reset provenance it consumed."""

    batch: RolloutBatch
    completed_episodes: tuple[CompletedEpisode, ...]
    reset_seeds: tuple[tuple[int, ...], ...]
    root_seed: int


@dataclass(frozen=True, slots=True)
class _EnvironmentTransitions:
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    communication_rejections: Tensor
    states: tuple[dict[str, Any], ...]
    observations: tuple[ObservationBatch, ...]


class SynchronousRolloutCollector:
    """Collect fixed-shape transitions without workers or hidden RNG state."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        environment_factory: EnvironmentFactory,
        num_environments: int,
        rollout_length: int,
        actor: SharedActor,
        critic: CentralizedCritic,
        root_seed: int,
        device: torch.device,
    ) -> None:
        if num_environments <= 0:
            raise ValueError("num_environments must be positive")
        if rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        self.environments = tuple(environment_factory() for _ in range(num_environments))
        if not self.environments:
            raise TrainingError("collector requires at least one environment")
        self.agent_order = tuple(self.environments[0].possible_agents)
        if not self.agent_order:
            raise TrainingError("environment possible_agents cannot be empty")
        for environment_id, environment in enumerate(self.environments):
            if tuple(environment.possible_agents) != self.agent_order:
                raise TrainingError(
                    f"environment {environment_id} has a different possible-agent ordering"
                )

        self.device = device
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_encoder = ActorObservationEncoder(
            self.environments[0].observation_space(self.agent_order[0])
        )
        self.critic_encoder = CentralStateEncoder(self.environments[0].state_space)
        self.seed_streams = ExperimentSeedStreams(root_seed)
        self.policy_generator = make_torch_generator(
            self.seed_streams.policy_sampling,
            device,
        )
        self.spec = RolloutSpec(
            rollout_length=rollout_length,
            num_environments=num_environments,
            agent_order=self.agent_order,
            actor_feature_dim=self.actor_encoder.input_dim,
            critic_feature_dim=self.critic_encoder.input_dim,
            move_action_count=self.actor_encoder.move_action_count,
            message_action_count=self.actor_encoder.message_action_count,
        )
        if self.actor.input_dim != self.spec.actor_feature_dim:
            raise TrainingError("actor input dimension does not match the observation adapter")
        if self.critic.input_dim != self.spec.critic_feature_dim:
            raise TrainingError("critic input dimension does not match the state adapter")
        if self.actor.move_action_count != self.spec.move_action_count:
            raise TrainingError("actor movement head does not match the environment action mask")
        if self.actor.message_action_count != self.spec.message_action_count:
            raise TrainingError("actor message head does not match the environment action mask")

        self._episode_indices = [0] * num_environments
        self._episode_lengths = [0] * num_environments
        self._transition_ids = [0] * num_environments
        self._episode_start = [True] * num_environments
        self._reset_seeds: list[list[int]] = [[] for _ in self.environments]
        self._observations = [
            self._reset_environment(environment_id) for environment_id in range(num_environments)
        ]

    def collect(self, *, deterministic: bool = False) -> RolloutCollection:
        """Collect one rollout, resetting each finished environment independently."""

        buffer = RolloutBuffer(self.spec, self.device)
        completed: list[CompletedEpisode] = []
        self.actor.eval()
        self.critic.eval()

        for _rollout_step in range(self.spec.rollout_length):
            active_agents = self._active_agent_mask()
            raw_observations = [
                self._observations[environment_id][agent]
                for environment_id in range(self.spec.num_environments)
                for agent in self.agent_order
            ]
            actor_batch = self.actor_encoder.encode(raw_observations, device=self.device)
            critic_input = self.critic_encoder.encode(
                [environment.state() for environment in self.environments],
                device=self.device,
            )
            with torch.no_grad():
                selection = select_actions(
                    self.actor,
                    actor_batch,
                    deterministic=deterministic,
                    generator=None if deterministic else self.policy_generator,
                )
                values = self.critic(critic_input)

            environment_ids = torch.arange(
                self.spec.num_environments,
                dtype=torch.int64,
                device=self.device,
            )
            transition_ids = torch.tensor(
                self._transition_ids,
                dtype=torch.int64,
                device=self.device,
            )
            episode_starts = torch.tensor(
                self._episode_start,
                dtype=torch.bool,
                device=self.device,
            )
            shape = (self.spec.num_environments, self.spec.num_agents)
            statistics = selection.statistics
            move_actions = statistics.move_actions.reshape(shape)
            message_actions = statistics.message_actions.reshape(shape)

            transitions = self._step_environments(move_actions, message_actions)

            with torch.no_grad():
                next_values = self.critic(
                    self.critic_encoder.encode(transitions.states, device=self.device)
                )
            next_values = torch.where(
                transitions.terminated,
                torch.zeros_like(next_values),
                next_values,
            )

            buffer.append(
                RolloutStep(
                    actor_features=selection.actor_input.features.reshape(
                        self.spec.num_environments,
                        self.spec.num_agents,
                        self.spec.actor_feature_dim,
                    ),
                    critic_features=critic_input.features,
                    move_action_masks=selection.move_action_masks.reshape(
                        self.spec.num_environments,
                        self.spec.num_agents,
                        self.spec.move_action_count,
                    ),
                    message_action_masks=selection.message_action_masks.reshape(
                        self.spec.num_environments,
                        self.spec.num_agents,
                        self.spec.message_action_count,
                    ),
                    move_actions=move_actions,
                    message_actions=message_actions,
                    move_log_probabilities=statistics.move_log_probabilities.reshape(shape),
                    message_log_probabilities=statistics.message_log_probabilities.reshape(shape),
                    joint_log_probabilities=statistics.joint_log_probabilities.reshape(shape),
                    rewards=transitions.rewards,
                    values=values,
                    next_values=next_values,
                    terminated=transitions.terminated,
                    truncated=transitions.truncated,
                    episode_starts=episode_starts,
                    active_agents=active_agents,
                    communication_rejections=transitions.communication_rejections,
                    environment_ids=environment_ids,
                    transition_ids=transition_ids,
                )
            )
            self._advance_environments(transitions, completed)

        return RolloutCollection(
            batch=buffer.as_batch(),
            completed_episodes=tuple(completed),
            reset_seeds=tuple(tuple(seeds) for seeds in self._reset_seeds),
            root_seed=self.seed_streams.root_seed,
        )

    def close(self) -> None:
        """Close every owned environment."""

        for environment in self.environments:
            environment.close()

    def checkpoint_state(self) -> dict[str, Any]:
        """Export every mutable collector stream at a rollout boundary."""

        return {
            "schema_version": 1,
            "episode_indices": list(self._episode_indices),
            "episode_lengths": list(self._episode_lengths),
            "transition_ids": list(self._transition_ids),
            "episode_start": list(self._episode_start),
            "reset_seeds": [list(seeds) for seeds in self._reset_seeds],
            "policy_generator_state": self.policy_generator.get_state().cpu(),
            "environments": [environment.checkpoint_state() for environment in self.environments],
        }

    def restore_checkpoint_state(self, raw_checkpoint: Mapping[str, Any]) -> None:
        """Restore a validated collector state without sampling or replaying actions."""

        expected_keys = {
            "schema_version",
            "episode_indices",
            "episode_lengths",
            "transition_ids",
            "episode_start",
            "reset_seeds",
            "policy_generator_state",
            "environments",
        }
        if set(raw_checkpoint) != expected_keys:
            raise TrainingError("collector checkpoint fields do not match schema version 1")
        if raw_checkpoint["schema_version"] != 1:
            raise TrainingError("unsupported collector checkpoint schema version")
        count = self.spec.num_environments
        episode_indices = self._checkpoint_int_list(
            "episode_indices", raw_checkpoint["episode_indices"], count
        )
        episode_lengths = self._checkpoint_int_list(
            "episode_lengths", raw_checkpoint["episode_lengths"], count
        )
        transition_ids = self._checkpoint_int_list(
            "transition_ids", raw_checkpoint["transition_ids"], count
        )
        episode_start = self._checkpoint_bool_list(
            "episode_start", raw_checkpoint["episode_start"], count
        )
        reset_seeds_raw = raw_checkpoint["reset_seeds"]
        if not isinstance(reset_seeds_raw, list) or len(reset_seeds_raw) != count:
            raise TrainingError("collector checkpoint reset_seeds has invalid shape")
        reset_seeds = [
            self._checkpoint_int_list(f"reset_seeds[{index}]", seeds, None)
            for index, seeds in enumerate(reset_seeds_raw)
        ]
        if any(not seeds for seeds in reset_seeds):
            raise TrainingError("collector checkpoint reset seed histories cannot be empty")
        environments_raw = raw_checkpoint["environments"]
        if not isinstance(environments_raw, list) or len(environments_raw) != count:
            raise TrainingError("collector checkpoint environment state count is invalid")
        if not all(isinstance(state, Mapping) for state in environments_raw):
            raise TrainingError("collector checkpoint environment states must be mappings")
        generator_state = raw_checkpoint["policy_generator_state"]
        if (
            not isinstance(generator_state, Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.ndim != 1
        ):
            raise TrainingError("collector checkpoint policy RNG state is invalid")

        observations: list[ObservationBatch] = []
        for environment, state in zip(self.environments, environments_raw, strict=True):
            observations.append(environment.restore_checkpoint_state(state))
        self._episode_indices = episode_indices
        self._episode_lengths = episode_lengths
        self._transition_ids = transition_ids
        self._episode_start = episode_start
        self._reset_seeds = reset_seeds
        self._observations = observations
        try:
            self.policy_generator.set_state(generator_state.cpu())
        except RuntimeError as exc:
            raise TrainingError("cannot restore collector policy RNG state") from exc

    @staticmethod
    def _checkpoint_int_list(
        name: str,
        raw: Any,
        expected_length: int | None,
    ) -> list[int]:
        if not isinstance(raw, list) or (
            expected_length is not None and len(raw) != expected_length
        ):
            raise TrainingError(f"collector checkpoint {name} has invalid shape")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw):
            raise TrainingError(f"collector checkpoint {name} must contain non-negative integers")
        return list(raw)

    @staticmethod
    def _checkpoint_bool_list(name: str, raw: Any, expected_length: int) -> list[bool]:
        if (
            not isinstance(raw, list)
            or len(raw) != expected_length
            or any(not isinstance(value, bool) for value in raw)
        ):
            raise TrainingError(f"collector checkpoint {name} must contain booleans")
        return list(raw)

    def _step_environments(
        self,
        move_actions: Tensor,
        message_actions: Tensor,
    ) -> _EnvironmentTransitions:
        rewards = torch.zeros(self.spec.num_environments, dtype=torch.float32, device=self.device)
        terminated = torch.zeros(self.spec.num_environments, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.spec.num_environments, dtype=torch.bool, device=self.device)
        rejections = torch.zeros(
            (self.spec.num_environments, self.spec.num_agents),
            dtype=torch.bool,
            device=self.device,
        )
        states: list[dict[str, Any]] = []
        observations_by_environment: list[ObservationBatch] = []
        for environment_id, environment in enumerate(self.environments):
            acting_agents = tuple(environment.agents)
            actions: ActionBatch = {}
            for agent in acting_agents:
                slot = self.agent_order.index(agent)
                actions[agent] = {
                    "move": int(move_actions[environment_id, slot].item()),
                    "message": int(message_actions[environment_id, slot].item()),
                }
            observations, reward_by_agent, terminations, truncations, infos = environment.step(
                actions
            )
            rewards[environment_id] = self._shared_reward(reward_by_agent)
            ended = not environment.agents
            episode_terminated = ended and any(terminations.values())
            episode_truncated = ended and any(truncations.values())
            if episode_terminated and episode_truncated:
                raise TrainingError(
                    f"environment {environment_id} ended as both terminated and truncated"
                )
            terminated[environment_id] = episode_terminated
            truncated[environment_id] = episode_truncated
            for agent in acting_agents:
                slot = self.agent_order.index(agent)
                rejections[environment_id, slot] = bool(
                    infos[agent].get("communication_rejected", False)
                )
            self._episode_lengths[environment_id] += 1
            self._transition_ids[environment_id] += 1
            states.append(environment.state())
            observations_by_environment.append(observations)
        return _EnvironmentTransitions(
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            communication_rejections=rejections,
            states=tuple(states),
            observations=tuple(observations_by_environment),
        )

    def _advance_environments(
        self,
        transitions: _EnvironmentTransitions,
        completed: list[CompletedEpisode],
    ) -> None:
        for environment_id in range(self.spec.num_environments):
            if bool(
                transitions.terminated[environment_id] or transitions.truncated[environment_id]
            ):
                completed.append(
                    CompletedEpisode(
                        environment_id=environment_id,
                        episode_index=self._episode_indices[environment_id],
                        reset_seed=self._reset_seeds[environment_id][-1],
                        length=self._episode_lengths[environment_id],
                        terminated=bool(transitions.terminated[environment_id]),
                        truncated=bool(transitions.truncated[environment_id]),
                    )
                )
                self._episode_indices[environment_id] += 1
                self._episode_lengths[environment_id] = 0
                self._observations[environment_id] = self._reset_environment(environment_id)
                self._episode_start[environment_id] = True
            else:
                self._update_slot_observations(
                    environment_id,
                    transitions.observations[environment_id],
                )
                self._episode_start[environment_id] = False

    def _reset_environment(self, environment_id: int) -> ObservationBatch:
        episode_index = self._episode_indices[environment_id]
        seed = self.seed_streams.environment_reset(environment_id, episode_index)
        observations, _infos = self.environments[environment_id].reset(seed=seed)
        if set(observations) != set(self.agent_order):
            raise TrainingError(
                f"environment {environment_id} reset did not return every possible agent"
            )
        self._reset_seeds[environment_id].append(seed)
        return observations

    def _update_slot_observations(
        self,
        environment_id: int,
        observations: ObservationBatch,
    ) -> None:
        unknown = set(observations) - set(self.agent_order)
        if unknown:
            raise TrainingError(
                f"environment {environment_id} returned unknown observation agents: "
                f"{sorted(unknown)}"
            )
        self._observations[environment_id].update(observations)

    def _active_agent_mask(self) -> Tensor:
        mask = torch.zeros(
            (self.spec.num_environments, self.spec.num_agents),
            dtype=torch.bool,
            device=self.device,
        )
        for environment_id, environment in enumerate(self.environments):
            for agent in environment.agents:
                try:
                    slot = self.agent_order.index(agent)
                except ValueError as exc:
                    raise TrainingError(
                        f"environment {environment_id} exposed unknown live agent {agent}"
                    ) from exc
                mask[environment_id, slot] = True
        if not bool(mask.any(dim=1).all()):
            raise TrainingError("collector encountered an environment with no live agents")
        return mask

    @staticmethod
    def _shared_reward(rewards: Mapping[AgentId, float]) -> float:
        if not rewards:
            raise TrainingError("environment returned no reward for an acting transition")
        values = tuple(float(reward) for reward in rewards.values())
        if not all(math.isfinite(value) for value in values):
            raise NumericalError("environment returned a NaN or infinite reward")
        if any(value != values[0] for value in values[1:]):
            raise TrainingError("v0.1 collector requires one shared team reward")
        return values[0]
