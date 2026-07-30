import numpy as np

from kovara9.config.models import EnvConfig
from kovara9.core.types import Position, WorldSnapshot
from kovara9.environments.grid_rescue.observations import (
    build_central_state,
    build_observation,
)


def _snapshot() -> WorldSnapshot:
    obstacles = np.zeros((7, 7), dtype=np.bool_)
    obstacles[0, 1] = True
    obstacles.flags.writeable = False
    return WorldSnapshot(
        width=7,
        height=7,
        obstacles=obstacles,
        agent_positions={"agent_0": Position(0, 0), "agent_1": Position(1, 1)},
        targets=frozenset({Position(0, 2), Position(6, 6)}),
        recovered_targets=frozenset({Position(0, 2)}),
        communication_budgets={"agent_0": 3, "agent_1": 2},
        latest_messages={"agent_0": 1, "agent_1": 2},
        step_count=4,
    )


def test_local_observation_pads_edges_and_hides_distant_target() -> None:
    config = EnvConfig(
        width=7,
        height=7,
        num_agents=2,
        obstacle_density=0.0,
        observation_radius=2,
        num_targets=2,
        max_steps=20,
    )
    observation = build_observation(_snapshot(), config, "agent_0", ("agent_0", "agent_1"))
    grid = observation["local_grid"]
    assert grid.shape == (5, 5, 5)
    assert grid[1, 0, 0] == 1
    assert grid[4, 3, 3] == 1
    assert grid[3, 2, 4] == 1
    assert int(grid[2].sum()) == 0
    assert observation["messages"].tolist() == [0, 2, 0, 0]
    assert int(observation["communication_budget"]) == 3


def test_central_state_contains_full_map_and_agent_slots() -> None:
    state = build_central_state(_snapshot(), ("agent_0", "agent_1"))
    assert state["global_grid"].shape == (7, 7, 7)
    assert state["global_grid"][1, 6, 6] == 1
    assert state["active_agents"].tolist() == [1, 1, 0, 0]
    assert state["communication_budgets"].tolist() == [3, 2, 0, 0]
    assert int(state["step_count"]) == 4
