"""Evaluation orchestration separated from policies, metrics, and artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from kovara9.agents.policy import Policy, PolicyTransitionInfo
from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.seeding import derive_seed
from kovara9.core.types import AgentId, Position, WorldSnapshot
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.evaluation.metrics import (
    aggregate_records,
    duplicated_exploration,
    exploration_coverage,
    team_efficiency,
)
from kovara9.evaluation.records import EpisodeRecord, EvaluationSummary

PolicyFactory = Callable[[], Policy]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One complete, in-memory evaluation suite."""

    records: tuple[EpisodeRecord, ...]
    summary: EvaluationSummary
    policy_parameters: Mapping[str, bool | float | int | str]


def _visible_reachable_cells(
    snapshot: WorldSnapshot,
    agent: AgentId,
    radius: int,
) -> set[Position]:
    center = snapshot.agent_positions[agent]
    cells: set[Position] = set()
    for row in range(max(0, center.row - radius), min(snapshot.height, center.row + radius + 1)):
        for col in range(
            max(0, center.col - radius),
            min(snapshot.width, center.col + radius + 1),
        ):
            position = Position(row, col)
            if not snapshot.obstacles[row, col]:
                cells.add(position)
    return cells


def run_episode(
    *,
    env_config: EnvConfig,
    seed: int,
    policy_factory: PolicyFactory,
) -> EpisodeRecord:
    """Run one episode and calculate metrics from factual simulator state."""

    env = GridRescueParallelEnv(env_config)
    observations, _infos = env.reset(seed=seed)
    policies = {agent: policy_factory() for agent in env.possible_agents}
    for slot, agent in enumerate(env.possible_agents):
        policies[agent].reset(
            agent_id=agent,
            observation_space=env.observation_space(agent),
            action_space=env.action_space(agent),
            seed=derive_seed(seed, "policy", slot),
        )

    observed = {agent: set[Position]() for agent in env.possible_agents}
    for agent in env.agents:
        observed[agent].update(
            _visible_reachable_cells(env.snapshot, agent, env_config.observation_radius)
        )
    agent_steps = 0
    messages = 0
    shared_return = 0.0
    success = False
    termination_reason = "time_limit"

    while env.agents:
        acting_agents = tuple(env.agents)
        actions = {agent: policies[agent].act(observations[agent]) for agent in acting_agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        agent_steps += len(acting_agents)
        messages += env.last_events.messages_sent
        if rewards:
            shared_return += next(iter(rewards.values()))
        for agent in acting_agents:
            policy_info = PolicyTransitionInfo(
                blocked=bool(infos[agent]["blocked"]),
                message_sent=bool(infos[agent]["message_sent"]),
                communication_rejected=bool(infos[agent]["communication_rejected"]),
            )
            policies[agent].observe_outcome(
                reward=rewards[agent],
                terminated=terminations[agent],
                truncated=truncations[agent],
                info=policy_info,
            )
        for agent in acting_agents:
            observed[agent].update(
                _visible_reachable_cells(env.snapshot, agent, env_config.observation_radius)
            )
        success = any(terminations.values())
        if success:
            termination_reason = "success"

    final_snapshot = env.snapshot
    reachable = int((~final_snapshot.obstacles).sum())
    record = EpisodeRecord(
        seed=seed,
        success=success,
        episode_length=final_snapshot.step_count,
        targets_recovered=len(final_snapshot.recovered_targets),
        total_targets=len(final_snapshot.targets),
        exploration_coverage=exploration_coverage(observed.values(), reachable),
        duplicated_exploration=duplicated_exploration(observed.values()),
        communication_messages=messages,
        messages_per_agent_step=messages / agent_steps if agent_steps else 0.0,
        team_efficiency=team_efficiency(len(final_snapshot.recovered_targets), agent_steps),
        shared_return=shared_return,
        termination_reason=termination_reason,
    )
    env.close()
    return record


def evaluate_policy(
    *,
    env_config: EnvConfig,
    evaluation_config: EvaluationConfig,
    policy_factory: PolicyFactory,
) -> EvaluationResult:
    """Evaluate a fresh independent policy team for every configured seed."""

    env_config = EnvConfig.model_validate(env_config.model_dump(mode="python", round_trip=True))
    evaluation_config = EvaluationConfig.model_validate(
        evaluation_config.model_dump(mode="python", round_trip=True)
    )
    probe = policy_factory()
    records = tuple(
        run_episode(env_config=env_config, seed=seed, policy_factory=policy_factory)
        for seed in evaluation_config.resolved_seeds
    )
    summary = aggregate_records(records, evaluation_config, probe.name)
    return EvaluationResult(
        records=records,
        summary=summary,
        policy_parameters=MappingProxyType(dict(probe.parameters)),
    )
