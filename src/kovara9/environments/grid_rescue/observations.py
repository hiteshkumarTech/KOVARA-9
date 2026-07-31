"""Construction of decentralized observations and centralized state."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np

from kovara9.config.models import EnvConfig
from kovara9.core.types import MAX_AGENTS, AgentId, Move, Position, WorldSnapshot

VISIBLE = 0
OBSTACLE = 1
TARGET = 2
RECOVERED_TARGET = 3
TEAMMATE = 4
LOCAL_CHANNELS = 5
GLOBAL_BASE_CHANNELS = 3


def build_observation(
    snapshot: WorldSnapshot,
    config: EnvConfig,
    agent: AgentId,
    possible_agents: tuple[AgentId, ...],
) -> dict[str, Any]:
    """Build one fixed-shape observation without global-state leakage."""

    radius = config.observation_radius
    diameter = 2 * radius + 1
    grid = np.zeros((LOCAL_CHANNELS, diameter, diameter), dtype=np.uint8)
    center = snapshot.agent_positions[agent]
    teammates_by_position = {
        position for teammate, position in snapshot.agent_positions.items() if teammate != agent
    }

    for local_row in range(diameter):
        for local_col in range(diameter):
            world = Position(
                center.row + local_row - radius,
                center.col + local_col - radius,
            )
            inside = 0 <= world.row < snapshot.height and 0 <= world.col < snapshot.width
            grid[VISIBLE, local_row, local_col] = 1
            if not inside or snapshot.obstacles[world.row, world.col]:
                grid[OBSTACLE, local_row, local_col] = 1
                continue
            if world in snapshot.targets and world not in snapshot.recovered_targets:
                grid[TARGET, local_row, local_col] = 1
            if world in snapshot.recovered_targets:
                grid[RECOVERED_TARGET, local_row, local_col] = 1
            if world in teammates_by_position:
                grid[TEAMMATE, local_row, local_col] = 1

    active_agents = np.zeros(MAX_AGENTS, dtype=np.int8)
    messages = np.zeros(MAX_AGENTS, dtype=np.int64)
    for slot, possible_agent in enumerate(possible_agents):
        active_agents[slot] = int(possible_agent in snapshot.agent_positions)
        if possible_agent != agent:
            messages[slot] = snapshot.latest_messages.get(possible_agent, 0)

    message_count = config.communication.vocabulary_size + 1
    if not config.communication.enabled:
        message_count = 1
    message_mask = np.zeros(message_count, dtype=np.int8)
    message_mask[0] = 1
    if snapshot.communication_budgets[agent] > 0:
        message_mask[:] = 1

    return {
        "local_grid": grid,
        "active_agents": active_agents,
        "messages": messages,
        "communication_budget": np.int64(snapshot.communication_budgets[agent]),
        "move_action_mask": np.ones(len(Move), dtype=np.int8),
        "message_action_mask": message_mask,
    }


def build_central_state(
    snapshot: WorldSnapshot,
    possible_agents: tuple[AgentId, ...],
    live_agents: Collection[AgentId] | None = None,
) -> dict[str, Any]:
    """Build the trainer/debug-only full state."""

    live_agent_set = set(possible_agents if live_agents is None else live_agents)
    channels = GLOBAL_BASE_CHANNELS + MAX_AGENTS
    grid = np.zeros((channels, snapshot.height, snapshot.width), dtype=np.uint8)
    grid[0] = snapshot.obstacles
    for target in snapshot.targets - snapshot.recovered_targets:
        grid[1, target.row, target.col] = 1
    for target in snapshot.recovered_targets:
        grid[2, target.row, target.col] = 1

    active_agents = np.zeros(MAX_AGENTS, dtype=np.int8)
    budgets = np.zeros(MAX_AGENTS, dtype=np.int64)
    messages = np.zeros(MAX_AGENTS, dtype=np.int64)
    for slot, agent in enumerate(possible_agents):
        position = snapshot.agent_positions.get(agent)
        if position is not None:
            active_agents[slot] = int(agent in live_agent_set)
            grid[GLOBAL_BASE_CHANNELS + slot, position.row, position.col] = 1
        budgets[slot] = snapshot.communication_budgets.get(agent, 0)
        messages[slot] = snapshot.latest_messages.get(agent, 0)
    return {
        "global_grid": grid,
        "active_agents": active_agents,
        "communication_budgets": budgets,
        "latest_messages": messages,
        "step_count": np.int64(snapshot.step_count),
    }
