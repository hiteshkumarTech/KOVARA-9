from collections import deque

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kovara9.config.models import EnvConfig, GenerationConfig
from kovara9.core.errors import GenerationError
from kovara9.core.seeding import make_rng
from kovara9.core.types import Position
from kovara9.environments.grid_rescue import generation
from kovara9.environments.grid_rescue.generation import generate_world


def _reachable(obstacles: np.ndarray, start: Position) -> set[Position]:
    visited = {start}
    queue = deque([start])
    height, width = obstacles.shape
    while queue:
        current = queue.popleft()
        for delta in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor = current.moved(*delta)
            if (
                0 <= neighbor.row < height
                and 0 <= neighbor.col < width
                and not obstacles[neighbor.row, neighbor.col]
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def test_generation_is_byte_reproducible(easy_config: EnvConfig) -> None:
    first = generate_world(easy_config, make_rng(91))
    second = generate_world(easy_config, make_rng(91))
    assert first.obstacles.tobytes() == second.obstacles.tobytes()
    assert first.agent_positions == second.agent_positions
    assert first.targets == second.targets


@given(
    size=st.integers(min_value=5, max_value=10),
    agents=st.integers(min_value=2, max_value=4),
    targets=st.integers(min_value=1, max_value=5),
    density=st.floats(min_value=0.0, max_value=0.25, allow_nan=False),
    seed=st.integers(min_value=0, max_value=100_000),
)
@settings(max_examples=20, deadline=None)
def test_generated_worlds_are_connected_and_unique(
    size: int,
    agents: int,
    targets: int,
    density: float,
    seed: int,
) -> None:
    config = EnvConfig(
        width=size,
        height=size,
        num_agents=agents,
        obstacle_density=density,
        observation_radius=2,
        num_targets=targets,
        max_steps=100,
    )
    state = generate_world(config, make_rng(seed))
    free = int((~state.obstacles).sum())
    reached = _reachable(state.obstacles, next(iter(state.agent_positions.values())))
    assert len(reached) == free
    occupied = set(state.agent_positions.values())
    assert len(occupied) == agents
    assert len(state.targets) == targets
    assert occupied.isdisjoint(state.targets)


def test_generation_failure_reports_parameters(
    easy_config: EnvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impossible_rng = np.random.default_rng(0)
    config = easy_config.model_copy(
        update={
            "obstacle_density": 0.45,
            "generation": GenerationConfig(max_attempts=1),
        }
    )
    monkeypatch.setattr(generation, "_connected_free_cells", lambda _obstacles: set())
    with pytest.raises(GenerationError, match="attempts"):
        generate_world(config, impossible_rng)
