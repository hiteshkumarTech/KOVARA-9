"""PettingZoo parallel environment for cooperative procedural rescue."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from kovara9.config.models import EnvConfig
from kovara9.core.errors import InvalidActionError
from kovara9.core.seeding import derive_seed, make_rng
from kovara9.core.types import MAX_AGENTS, AgentAction, AgentId, Move, Position, StepEvents
from kovara9.environments.grid_rescue.generation import agent_slots, generate_world
from kovara9.environments.grid_rescue.observations import (
    GLOBAL_BASE_CHANNELS,
    LOCAL_CHANNELS,
    build_central_state,
    build_observation,
)
from kovara9.environments.grid_rescue.state import WorldState
from kovara9.rendering.ansi import AnsiRenderer
from kovara9.rendering.rgb_array import RgbArrayRenderer


class GridRescueParallelEnv(ParallelEnv):  # type: ignore[misc]
    """Simultaneous, partially observable cooperative rescue environment."""

    metadata: ClassVar[dict[str, Any]] = {
        "name": "KovaraGridRescue-v0",
        "render_modes": ["ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, config: EnvConfig, render_mode: str | None = None) -> None:
        if render_mode not in {None, "ansi", "rgb_array"}:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        config = EnvConfig.model_validate(config.model_dump(mode="python", round_trip=True))
        self.config = config
        self.render_mode = render_mode
        self.possible_agents = list(agent_slots(config.num_agents))
        self.agents: list[AgentId] = []
        self._possible_agents_tuple = tuple(self.possible_agents)
        self._episode_counter = 0
        self._state: WorldState | None = None
        self.last_events = StepEvents((), 0, ())

        message_count = config.communication.vocabulary_size + 1
        if not config.communication.enabled:
            message_count = 1
        diameter = 2 * config.observation_radius + 1
        observation_space = spaces.Dict(
            {
                "local_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(LOCAL_CHANNELS, diameter, diameter),
                    dtype=np.uint8,
                ),
                "active_agents": spaces.MultiBinary(MAX_AGENTS),
                "messages": spaces.MultiDiscrete(
                    np.full(MAX_AGENTS, message_count, dtype=np.int64)
                ),
                "communication_budget": spaces.Discrete(config.communication.budget_per_agent + 1),
                "move_action_mask": spaces.MultiBinary(len(Move)),
                "message_action_mask": spaces.MultiBinary(message_count),
            }
        )
        action_space = spaces.Dict(
            {
                "move": spaces.Discrete(len(Move)),
                "message": spaces.Discrete(message_count),
            }
        )
        self.observation_spaces = dict.fromkeys(self.possible_agents, observation_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, action_space)
        self.state_space = spaces.Dict(
            {
                "global_grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        GLOBAL_BASE_CHANNELS + MAX_AGENTS,
                        config.height,
                        config.width,
                    ),
                    dtype=np.uint8,
                ),
                "active_agents": spaces.MultiBinary(MAX_AGENTS),
                "communication_budgets": spaces.Box(
                    low=0,
                    high=config.communication.budget_per_agent,
                    shape=(MAX_AGENTS,),
                    dtype=np.int64,
                ),
                "latest_messages": spaces.MultiDiscrete(
                    np.full(MAX_AGENTS, message_count, dtype=np.int64)
                ),
                "step_count": spaces.Discrete(config.max_steps + 1),
            }
        )

    @property
    def snapshot(self) -> Any:
        """Return an immutable defensive snapshot of the current world."""

        return self._require_state().snapshot()

    def observation_space(self, agent: AgentId) -> spaces.Space[Any]:
        """Return one agent's observation space."""

        return self.observation_spaces[agent]

    def action_space(self, agent: AgentId) -> spaces.Space[Any]:
        """Return one agent's action space."""

        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentId, dict[str, Any]], dict[AgentId, dict[str, Any]]]:
        """Generate a new deterministic episode."""

        unused_options = sorted(options) if options else []
        episode_seed = (
            seed if seed is not None else derive_seed(0, "environment-reset", self._episode_counter)
        )
        self._episode_counter += 1
        self._state = generate_world(self.config, make_rng(episode_seed))
        self.agents = list(self.possible_agents)
        self.last_events = StepEvents((), 0, ())
        observations = self._observations()
        infos = {
            agent: {
                "seed": episode_seed,
                "environment_id": self.metadata["name"],
                "unused_reset_options": unused_options,
            }
            for agent in self.agents
        }
        return observations, infos

    def step(
        self,
        actions: Mapping[AgentId, Mapping[str, Any] | AgentAction],
    ) -> tuple[
        dict[AgentId, dict[str, Any]],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, dict[str, Any]],
    ]:
        """Apply one validated joint action and return the parallel API tuple."""

        state = self._require_state()
        acting_agents = tuple(self.agents)
        if not acting_agents:
            raise RuntimeError("step called after the episode ended; call reset()")
        if set(actions) != set(acting_agents):
            missing = sorted(set(acting_agents) - set(actions))
            extra = sorted(set(actions) - set(acting_agents))
            raise InvalidActionError(f"joint action mismatch: missing={missing}, extra={extra}")

        parsed = {agent: self._parse_action(agent, actions[agent]) for agent in acting_agents}
        destinations, blocked = self._resolve_movements(state, parsed)
        state.agent_positions.update(destinations)

        rejected_message_agents = {
            agent
            for agent, action in parsed.items()
            if action.message != 0 and state.communication_budgets[agent] <= 0
        }
        accepted_messages = {
            agent: 0 if agent in rejected_message_agents else action.message
            for agent, action in parsed.items()
        }
        messages_sent = sum(message != 0 for message in accepted_messages.values())
        for agent in parsed:
            if accepted_messages[agent]:
                state.communication_budgets[agent] -= 1
        state.latest_messages = accepted_messages

        newly_recovered = tuple(
            sorted(
                {
                    position
                    for position in state.agent_positions.values()
                    if position in state.targets and position not in state.recovered_targets
                }
            )
        )
        state.recovered_targets.update(newly_recovered)
        state.step_count += 1

        success = state.recovered_targets == state.targets
        truncated = state.step_count >= self.config.max_steps and not success
        team_reward = (
            len(newly_recovered) * self.config.reward.target_recovery
            + self.config.reward.step_penalty
            + messages_sent * self.config.reward.message_penalty
            + (self.config.reward.success_bonus if success else 0.0)
        )
        self.last_events = StepEvents(
            newly_recovered,
            messages_sent,
            tuple(sorted(blocked)),
            tuple(sorted(rejected_message_agents)),
        )

        rewards = dict.fromkeys(acting_agents, team_reward)
        terminations = dict.fromkeys(acting_agents, success)
        truncations = dict.fromkeys(acting_agents, truncated)
        infos = {
            agent: {
                "blocked": agent in blocked,
                "message_sent": accepted_messages[agent] != 0,
                "communication_rejected": agent in rejected_message_agents,
            }
            for agent in acting_agents
        }
        if success or truncated:
            observations = self._observations()
            self.agents = []
        else:
            observations = self._observations()
        return observations, rewards, terminations, truncations, infos

    def state(self) -> dict[str, Any]:
        """Return full state for future centralized training and diagnostics."""

        return build_central_state(
            self._require_state().snapshot(),
            self._possible_agents_tuple,
            self.agents,
        )

    def render(self) -> str | np.ndarray | None:
        """Render the current snapshot without changing simulator state."""

        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return AnsiRenderer().render(self.snapshot)
        return RgbArrayRenderer().render(self.snapshot)

    def close(self) -> None:
        """Release renderer resources; Phase 0 renderers are resource-free."""

    def _observations(self) -> dict[AgentId, dict[str, Any]]:
        snapshot = self._require_state().snapshot()
        return {
            agent: build_observation(
                snapshot,
                self.config,
                agent,
                self._possible_agents_tuple,
            )
            for agent in self.agents
        }

    def _parse_action(
        self,
        agent: AgentId,
        raw_action: Mapping[str, Any] | AgentAction,
    ) -> AgentAction:
        if isinstance(raw_action, AgentAction):
            raw_move: Any = raw_action.move
            raw_message: Any = raw_action.message
        else:
            if not isinstance(raw_action, Mapping):
                raise InvalidActionError(f"invalid action for {agent}: {raw_action}")
            if set(raw_action) != {"move", "message"}:
                raise InvalidActionError(
                    f"{agent} action must contain exactly 'move' and 'message'"
                )
            raw_move = raw_action["move"]
            raw_message = raw_action["message"]
        if (
            isinstance(raw_move, bool)
            or not isinstance(raw_move, Integral)
            or isinstance(raw_message, bool)
            or not isinstance(raw_message, Integral)
        ):
            raise InvalidActionError(
                f"{agent} move and message must be integral action values: {raw_action}"
            )
        try:
            action = AgentAction(move=Move(int(raw_move)), message=int(raw_message))
        except ValueError as exc:
            raise InvalidActionError(f"invalid action for {agent}: {raw_action}") from exc
        max_message = (
            self.config.communication.vocabulary_size if self.config.communication.enabled else 0
        )
        if not 0 <= action.message <= max_message:
            raise InvalidActionError(
                f"{agent} message {action.message} is outside [0, {max_message}]"
            )
        return action

    @staticmethod
    def _resolve_movements(
        state: WorldState,
        actions: Mapping[AgentId, AgentAction],
    ) -> tuple[dict[AgentId, Position], set[AgentId]]:
        origins = dict(state.agent_positions)
        intended: dict[AgentId, Position] = {}
        blocked: set[AgentId] = set()
        for agent, action in actions.items():
            row_delta, col_delta = action.move.delta
            destination = origins[agent].moved(row_delta, col_delta)
            valid = (
                0 <= destination.row < state.height
                and 0 <= destination.col < state.width
                and not state.obstacles[destination.row, destination.col]
            )
            if action.move is Move.STAY or not valid:
                intended[agent] = origins[agent]
                if action.move is not Move.STAY:
                    blocked.add(agent)
            else:
                intended[agent] = destination

        claims: dict[Position, list[AgentId]] = {}
        for agent, destination in intended.items():
            if destination != origins[agent]:
                claims.setdefault(destination, []).append(agent)
        candidates = {
            agent: destination
            for agent, destination in intended.items()
            if destination != origins[agent] and len(claims[destination]) == 1
        }
        for claimants in claims.values():
            if len(claimants) > 1:
                blocked.update(claimants)

        occupant_by_origin = {position: agent for agent, position in origins.items()}
        memo: dict[AgentId, bool] = {}

        def can_move(agent: AgentId, path: frozenset[AgentId]) -> bool:
            cached = memo.get(agent)
            if cached is not None:
                return cached
            occupant = occupant_by_origin.get(candidates[agent])
            if occupant is None:
                memo[agent] = True
                return True
            if occupant not in candidates or occupant in path:
                memo[agent] = False
                return False
            result = can_move(occupant, path | {agent})
            memo[agent] = result
            return result

        destinations = dict(origins)
        for agent in sorted(candidates):
            if can_move(agent, frozenset()):
                destinations[agent] = candidates[agent]
            else:
                blocked.add(agent)
        return destinations, blocked

    def _require_state(self) -> WorldState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state
