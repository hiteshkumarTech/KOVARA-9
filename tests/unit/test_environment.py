import numpy as np
import pytest

from kovara9.config.models import CommunicationConfig, EnvConfig
from kovara9.core.errors import InvalidActionError
from kovara9.core.types import AgentAction, Move, Position
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.environments.grid_rescue.state import WorldState


def _state(
    positions: dict[str, Position],
    *,
    width: int = 5,
    height: int = 5,
) -> WorldState:
    return WorldState(
        width=width,
        height=height,
        obstacles=np.zeros((height, width), dtype=np.bool_),
        agent_positions=positions,
        targets={Position(4, 4)},
        recovered_targets=set(),
        communication_budgets=dict.fromkeys(positions, 2),
        latest_messages=dict.fromkeys(positions, 0),
    )


def _actions(**moves: Move) -> dict[str, AgentAction]:
    return {agent: AgentAction(move) for agent, move in moves.items()}


def test_chain_moves_into_empty_cell() -> None:
    state = _state({"agent_0": Position(1, 0), "agent_1": Position(1, 1)})
    result, blocked = GridRescueParallelEnv._resolve_movements(
        state, _actions(agent_0=Move.EAST, agent_1=Move.EAST)
    )
    assert result == {"agent_0": Position(1, 1), "agent_1": Position(1, 2)}
    assert not blocked


@pytest.mark.parametrize(
    ("positions", "moves"),
    [
        (
            {"agent_0": Position(1, 0), "agent_1": Position(1, 1)},
            {"agent_0": Move.EAST, "agent_1": Move.WEST},
        ),
        (
            {"agent_0": Position(1, 0), "agent_1": Position(1, 2)},
            {"agent_0": Move.EAST, "agent_1": Move.WEST},
        ),
    ],
)
def test_swaps_and_contested_destinations_fail(
    positions: dict[str, Position],
    moves: dict[str, Move],
) -> None:
    result, blocked = GridRescueParallelEnv._resolve_movements(_state(positions), _actions(**moves))
    assert result == positions
    assert blocked == set(positions)


def test_three_agent_collision_cycle_with_contested_leg_is_blocked_atomically() -> None:
    positions = {
        "agent_0": Position(1, 0),
        "agent_1": Position(1, 1),
        "agent_2": Position(1, 2),
    }
    result, blocked = GridRescueParallelEnv._resolve_movements(
        _state(positions),
        _actions(agent_0=Move.EAST, agent_1=Move.WEST, agent_2=Move.WEST),
    )
    assert result == positions
    assert blocked == set(positions)


def test_four_agent_movement_cycle_is_blocked_atomically() -> None:
    positions = {
        "agent_0": Position(1, 1),
        "agent_1": Position(1, 2),
        "agent_2": Position(2, 2),
        "agent_3": Position(2, 1),
    }
    result, blocked = GridRescueParallelEnv._resolve_movements(
        _state(positions),
        _actions(
            agent_0=Move.EAST,
            agent_1=Move.SOUTH,
            agent_2=Move.WEST,
            agent_3=Move.NORTH,
        ),
    )
    assert result == positions
    assert blocked == set(positions)


def test_blocked_movement_cascades_through_occupied_chain() -> None:
    positions = {
        "agent_0": Position(1, 0),
        "agent_1": Position(1, 1),
        "agent_2": Position(1, 2),
    }
    result, blocked = GridRescueParallelEnv._resolve_movements(
        _state(positions),
        _actions(agent_0=Move.EAST, agent_1=Move.EAST, agent_2=Move.STAY),
    )
    assert result == positions
    assert blocked == {"agent_0", "agent_1"}


def test_joint_action_dictionary_order_does_not_change_transition() -> None:
    positions = {"agent_0": Position(1, 0), "agent_1": Position(1, 1)}
    forward = _actions(agent_0=Move.EAST, agent_1=Move.EAST)
    reverse = dict(reversed(tuple(forward.items())))
    first = GridRescueParallelEnv._resolve_movements(_state(positions), forward)
    second = GridRescueParallelEnv._resolve_movements(_state(positions), reverse)
    assert first == second


def test_boundaries_walls_and_stationary_occupants_block() -> None:
    state = _state({"agent_0": Position(0, 0), "agent_1": Position(0, 1)})
    state.obstacles[1, 0] = True
    result, blocked = GridRescueParallelEnv._resolve_movements(
        state, _actions(agent_0=Move.SOUTH, agent_1=Move.WEST)
    )
    assert result == state.agent_positions
    assert blocked == {"agent_0", "agent_1"}


def test_reset_spaces_step_and_defensive_snapshot(easy_config: EnvConfig) -> None:
    env = GridRescueParallelEnv(easy_config, render_mode="rgb_array")
    observations, infos = env.reset(seed=12)
    assert set(observations) == set(env.possible_agents)
    assert all(env.observation_space(a).contains(observations[a]) for a in env.agents)
    assert all(
        observation["message_action_mask"].tolist() == [1, 1, 1, 1, 1]
        for observation in observations.values()
    )
    assert env.state_space.contains(env.state())
    assert all(infos[agent]["seed"] == 12 for agent in env.agents)
    assert env.render().shape == (8, 8, 3)
    snapshot = env.snapshot
    with pytest.raises(ValueError, match="read-only"):
        snapshot.obstacles[0, 0] = False
    actions = {agent: {"move": 0, "message": 0} for agent in env.agents}
    next_observations, rewards, terminated, truncated, _ = env.step(actions)
    assert set(next_observations) == set(env.agents)
    assert len(set(rewards.values())) == 1
    assert not any(terminated.values())
    assert not any(truncated.values())
    env.close()


def test_terminal_recovery_and_post_terminal_step(easy_config: EnvConfig) -> None:
    config = easy_config.model_copy(update={"max_steps": 5})
    env = GridRescueParallelEnv(config)
    env.reset(seed=1)
    state = _state(
        {"agent_0": Position(1, 0), "agent_1": Position(4, 0)},
        width=config.width,
        height=config.height,
    )
    state.targets = {Position(1, 1)}
    state.communication_budgets = {"agent_0": 2, "agent_1": 2}
    state.latest_messages = {"agent_0": 0, "agent_1": 0}
    env._state = state
    observations, rewards, terminated, truncated, infos = env.step(
        {
            "agent_0": {"move": int(Move.EAST), "message": 1},
            "agent_1": {"move": int(Move.STAY), "message": 0},
        }
    )
    assert set(observations) == {"agent_0", "agent_1"}
    assert all(terminated.values())
    assert not any(truncated.values())
    assert env.last_events.recovered_targets == (Position(1, 1),)
    assert infos["agent_0"] == {
        "blocked": False,
        "message_sent": True,
        "communication_rejected": False,
    }
    assert rewards["agent_0"] > 5
    assert env.state()["active_agents"].tolist() == [0, 0, 0, 0]
    assert env.state_space.contains(env.state())
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step({})


def test_invalid_actions_and_budget_rejection_are_explicit(easy_config: EnvConfig) -> None:
    env = GridRescueParallelEnv(easy_config)
    with pytest.raises(RuntimeError, match="not been reset"):
        env.state()
    env.reset(seed=1)
    with pytest.raises(InvalidActionError, match="mismatch"):
        env.step({})
    actions = {agent: {"move": 0, "message": 0} for agent in env.agents}
    actions["agent_0"] = {"move": 99, "message": 0}
    with pytest.raises(InvalidActionError, match="invalid action"):
        env.step(actions)
    actions["agent_0"] = {"move": 0}
    with pytest.raises(InvalidActionError, match="exactly"):
        env.step(actions)
    actions["agent_0"] = {"move": 0, "message": 99}
    with pytest.raises(InvalidActionError, match="outside"):
        env.step(actions)
    env._state.communication_budgets["agent_0"] = 0
    actions["agent_0"] = {"move": 0, "message": 1}
    observations, _rewards, _terminated, _truncated, infos = env.step(actions)
    assert infos["agent_0"]["communication_rejected"] is True
    assert infos["agent_0"]["message_sent"] is False
    assert env.last_events.rejected_message_agents == ("agent_0",)
    assert env.last_events.messages_sent == 0
    assert env.snapshot.communication_budgets["agent_0"] == 0
    assert env.snapshot.latest_messages["agent_0"] == 0
    assert observations["agent_0"]["message_action_mask"].tolist() == [1, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("move", 1.9),
        ("move", True),
        ("move", "1"),
        ("message", 1.0),
        ("message", False),
        ("message", "1"),
    ],
)
def test_non_integral_action_values_are_rejected(
    easy_config: EnvConfig,
    field: str,
    value: object,
) -> None:
    env = GridRescueParallelEnv(easy_config)
    env.reset(seed=1)
    actions: dict[str, dict[str, object]] = {
        agent: {"move": 0, "message": 0} for agent in env.agents
    }
    actions["agent_0"][field] = value
    with pytest.raises(InvalidActionError, match="integral action values"):
        env.step(actions)


def test_truncation_and_default_reset_seed(easy_config: EnvConfig) -> None:
    config = easy_config.model_copy(update={"max_steps": 1})
    env = GridRescueParallelEnv(config)
    first, first_infos = env.reset()
    assert first
    first_seed = first_infos["agent_0"]["seed"]
    actions = {agent: {"move": 0, "message": 0} for agent in env.agents}
    observations, _rewards, terminated, truncated, _infos = env.step(actions)
    assert set(observations) == {"agent_0", "agent_1"}
    assert not any(terminated.values())
    assert all(truncated.values())
    _, second_infos = env.reset()
    assert second_infos["agent_0"]["seed"] != first_seed


def test_constructor_and_reset_options_validate(easy_config: EnvConfig) -> None:
    with pytest.raises(ValueError, match="render_mode"):
        GridRescueParallelEnv(easy_config, render_mode="human")
    env = GridRescueParallelEnv(easy_config)
    _, infos = env.reset(options={"unknown": True})
    assert infos["agent_0"]["unused_reset_options"] == ["unknown"]
    assert env.render() is None


def test_communication_can_be_disabled(easy_config: EnvConfig) -> None:
    config = easy_config.model_copy(
        update={"communication": CommunicationConfig(enabled=False, budget_per_agent=0)}
    )
    env = GridRescueParallelEnv(config)
    observations, _ = env.reset(seed=2)
    assert all(env.action_space(agent).contains({"move": 0, "message": 0}) for agent in env.agents)
    assert all(int(obs["communication_budget"]) == 0 for obs in observations.values())
