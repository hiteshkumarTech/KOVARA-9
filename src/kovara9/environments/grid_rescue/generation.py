"""Deterministic procedural generation with solvability guarantees."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from kovara9.config.models import EnvConfig
from kovara9.core.errors import GenerationError
from kovara9.core.types import AgentId, Position
from kovara9.environments.grid_rescue.state import WorldState


def _connected_free_cells(obstacles: NDArray[np.bool_]) -> set[Position]:
    free_indices = np.argwhere(~obstacles)
    if free_indices.size == 0:
        return set()
    start = Position(int(free_indices[0, 0]), int(free_indices[0, 1]))
    visited = {start}
    frontier = deque([start])
    height, width = obstacles.shape
    while frontier:
        current = frontier.popleft()
        for row_delta, col_delta in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            neighbor = current.moved(row_delta, col_delta)
            if (
                0 <= neighbor.row < height
                and 0 <= neighbor.col < width
                and not obstacles[neighbor.row, neighbor.col]
                and neighbor not in visited
            ):
                visited.add(neighbor)
                frontier.append(neighbor)
    return visited


def generate_world(config: EnvConfig, rng: np.random.Generator) -> WorldState:
    """Generate a connected grid with unique reachable spawns and targets."""

    cell_count = config.width * config.height
    obstacle_count = round(cell_count * config.obstacle_density)
    required_free = config.num_agents + config.num_targets
    for _attempt in range(config.generation.max_attempts):
        obstacles = np.zeros((config.height, config.width), dtype=np.bool_)
        if obstacle_count:
            flat_obstacles = rng.choice(cell_count, size=obstacle_count, replace=False)
            obstacles.flat[flat_obstacles] = True
        connected = _connected_free_cells(obstacles)
        if len(connected) != cell_count - obstacle_count or len(connected) < required_free:
            continue

        ordered_free = sorted(connected)
        selected_indices = rng.choice(len(ordered_free), size=required_free, replace=False)
        selected = [ordered_free[int(index)] for index in selected_indices]
        agent_positions = {f"agent_{index}": selected[index] for index in range(config.num_agents)}
        targets = set(selected[config.num_agents :])
        return WorldState(
            width=config.width,
            height=config.height,
            obstacles=obstacles,
            agent_positions=agent_positions,
            targets=targets,
            recovered_targets=set(),
            communication_budgets=dict.fromkeys(
                agent_positions,
                config.communication.budget_per_agent,
            ),
            latest_messages=dict.fromkeys(agent_positions, 0),
        )
    raise GenerationError(
        "failed to generate a connected world "
        f"after {config.generation.max_attempts} attempts "
        f"(size={config.width}x{config.height}, density={config.obstacle_density})"
    )


def reachable_cells(state: WorldState) -> set[Position]:
    """Return all reachable cells; generated worlds guarantee one component."""

    return _connected_free_cells(state.obstacles)


def agent_slots(num_agents: int) -> tuple[AgentId, ...]:
    """Return stable agent identifiers."""

    return tuple(f"agent_{index}" for index in range(num_agents))
