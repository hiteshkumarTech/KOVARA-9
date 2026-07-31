from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Space

from kovara9.agents.random import RandomPolicy
from kovara9.config.models import CommunicationConfig, EnvConfig
from kovara9.core.types import AgentId, Move, Position
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.environments.grid_rescue.state import WorldState
from kovara9.evaluation import runner
from kovara9.evaluation.runner import run_episode


class _TerminalMetricEnvironment(GridRescueParallelEnv):
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentId, dict[str, Any]], dict[AgentId, dict[str, Any]]]:
        _observations, infos = super().reset(seed=seed, options=options)
        positions = {"agent_0": Position(2, 2), "agent_1": Position(2, 3)}
        self._state = WorldState(
            width=5,
            height=5,
            obstacles=np.zeros((5, 5), dtype=np.bool_),
            agent_positions=positions,
            targets={Position(4, 4)},
            recovered_targets=set(),
            communication_budgets=dict.fromkeys(positions, 0),
            latest_messages=dict.fromkeys(positions, 0),
        )
        return self._observations(), infos


class _TerminalMovementPolicy:
    def __init__(self) -> None:
        self._agent_id = ""

    @property
    def name(self) -> str:
        return "terminal-movement"

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
        del observation_space, action_space, seed
        self._agent_id = agent_id

    def act(self, observation: dict[str, Any]) -> dict[str, int]:
        del observation
        move = Move.WEST if self._agent_id == "agent_0" else Move.STAY
        return {"move": int(move), "message": 0}

    def observe_outcome(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        del reward, terminated, truncated, info


@pytest.fixture
def terminal_metric_config() -> EnvConfig:
    return EnvConfig(
        width=5,
        height=5,
        num_agents=2,
        obstacle_density=0.0,
        observation_radius=1,
        num_targets=1,
        max_steps=1,
        communication=CommunicationConfig(enabled=False, budget_per_agent=0),
    )


def test_terminal_movement_is_included_in_exploration_coverage(
    monkeypatch: pytest.MonkeyPatch,
    terminal_metric_config: EnvConfig,
) -> None:
    monkeypatch.setattr(runner, "GridRescueParallelEnv", _TerminalMetricEnvironment)
    record = run_episode(
        env_config=terminal_metric_config,
        seed=20_000,
        policy_factory=_TerminalMovementPolicy,
    )
    assert record.exploration_coverage == pytest.approx(15 / 25)


def test_terminal_movement_is_included_in_duplicated_exploration(
    monkeypatch: pytest.MonkeyPatch,
    terminal_metric_config: EnvConfig,
) -> None:
    monkeypatch.setattr(runner, "GridRescueParallelEnv", _TerminalMetricEnvironment)
    record = run_episode(
        env_config=terminal_metric_config,
        seed=20_000,
        policy_factory=_TerminalMovementPolicy,
    )
    assert record.duplicated_exploration == pytest.approx(6 / 21)


def test_baseline_policy_outcomes_exclude_team_global_information(
    monkeypatch: pytest.MonkeyPatch,
    easy_config: EnvConfig,
) -> None:
    received_infos: list[dict[str, Any]] = []

    def record_outcome(
        self: RandomPolicy,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        del self, reward, terminated, truncated
        received_infos.append(info)

    monkeypatch.setattr(RandomPolicy, "observe_outcome", record_outcome)
    run_episode(
        env_config=easy_config.model_copy(update={"max_steps": 1}),
        seed=20_000,
        policy_factory=RandomPolicy,
    )
    assert received_infos
    assert all(
        set(info) == {"blocked", "message_sent", "communication_rejected"}
        for info in received_infos
    )
    forbidden = {"newly_recovered", "messages_sent", "success"}
    assert all(forbidden.isdisjoint(info) for info in received_infos)
