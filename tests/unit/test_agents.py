import numpy as np
import pytest
from gymnasium import spaces

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.random import RandomPolicy
from kovara9.core.types import Move


def _spaces() -> tuple[spaces.Dict, spaces.Dict]:
    observation = spaces.Dict(
        {
            "local_grid": spaces.Box(0, 1, (5, 5, 5), dtype=np.uint8),
            "communication_budget": spaces.Discrete(5),
        }
    )
    action = spaces.Dict({"move": spaces.Discrete(5), "message": spaces.Discrete(5)})
    return observation, action


def _observation() -> dict[str, object]:
    grid = np.zeros((5, 5, 5), dtype=np.uint8)
    return {"local_grid": grid, "communication_budget": np.int64(4)}


def test_random_policy_is_seeded_valid_and_budget_aware() -> None:
    observation_space, action_space = _spaces()
    first = RandomPolicy(message_probability=1.0)
    second = RandomPolicy(message_probability=1.0)
    for policy in (first, second):
        policy.reset(
            agent_id="agent_0",
            observation_space=observation_space,
            action_space=action_space,
            seed=45,
        )
    actions_first = [first.act(_observation()) for _ in range(10)]
    actions_second = [second.act(_observation()) for _ in range(10)]
    assert actions_first == actions_second
    assert all(action_space.contains(action) for action in actions_first)
    assert all(action["message"] > 0 for action in actions_first)
    no_budget = {**_observation(), "communication_budget": np.int64(0)}
    assert first.act(no_budget)["message"] == 0
    first.observe_outcome(reward=0, terminated=False, truncated=False, info={})


def test_random_policy_validates_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RandomPolicy(message_probability=2)
    policy = RandomPolicy()
    with pytest.raises(TypeError):
        policy.reset(
            agent_id="agent_0",
            observation_space=spaces.Discrete(1),
            action_space=spaces.Discrete(1),
            seed=0,
        )


def test_frontier_moves_toward_visible_target_and_sends_token() -> None:
    observation_space, action_space = _spaces()
    policy = FrontierPolicy()
    policy.reset(
        agent_id="agent_0",
        observation_space=observation_space,
        action_space=action_space,
        seed=1,
    )
    observation = _observation()
    grid = observation["local_grid"]
    assert isinstance(grid, np.ndarray)
    grid[2, 2, 4] = 1
    action = policy.act(observation)
    assert action == {"move": int(Move.EAST), "message": 1}
    policy.observe_outcome(
        reward=0,
        terminated=False,
        truncated=False,
        info={"blocked": False},
    )


def test_frontier_avoids_obstacles_and_handles_no_moves() -> None:
    observation_space, action_space = _spaces()
    policy = FrontierPolicy()
    policy.reset(
        agent_id="agent_0",
        observation_space=observation_space,
        action_space=action_space,
        seed=1,
    )
    observation = _observation()
    grid = observation["local_grid"]
    assert isinstance(grid, np.ndarray)
    grid[1, 1, 2] = 1
    grid[1, 2, 3] = 1
    grid[1, 3, 2] = 1
    grid[1, 2, 1] = 1
    assert policy.act(observation)["move"] == int(Move.STAY)
    policy.observe_outcome(
        reward=0,
        terminated=False,
        truncated=True,
        info={"blocked": True},
    )
