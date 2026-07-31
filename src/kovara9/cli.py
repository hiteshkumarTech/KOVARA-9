"""Command-line entry point for simulation and evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import structlog
import typer

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.policy import Policy, PolicyTransitionInfo
from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import (
    load_comparison_environment_configs,
    load_environment_config,
    load_evaluation_config,
)
from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import ConfigurationError, KovaraError
from kovara9.core.seeding import derive_seed
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.evaluation.runner import evaluate_policy
from kovara9.reporting.artifacts import ArtifactWriter
from kovara9.reporting.summaries import comparison_summary

app = typer.Typer(
    name="kovara9",
    help="KOVARA-9 cooperative embodied multi-agent research tools.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Validate research configuration.")
environment_app = typer.Typer(help="Run and inspect environments.")
app.add_typer(config_app, name="config")
app.add_typer(environment_app, name="env")

PolicyName = Annotated[str, typer.Option(help="Baseline policy: random or frontier.")]


def _configure_logging(json_logs: bool) -> None:
    processors: list[Any] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(processors=processors)


@app.callback()
def main(
    json_logs: Annotated[
        bool,
        typer.Option("--json-logs", help="Emit structured JSON logs."),
    ] = False,
) -> None:
    """Configure process-wide logging."""

    _configure_logging(json_logs)


def _policy_factory(name: str) -> type[Policy]:
    normalized = name.casefold()
    if normalized == "random":
        return RandomPolicy
    if normalized == "frontier":
        return FrontierPolicy
    raise typer.BadParameter("agent must be 'random' or 'frontier'")


def _abort(exc: KovaraError) -> None:
    structlog.get_logger().error("command_failed", error=str(exc))
    raise typer.Exit(code=2) from exc


@config_app.command("validate")
def validate_config(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate an environment or evaluation YAML configuration."""

    config: EnvConfig | EvaluationConfig
    try:
        try:
            config = load_environment_config(path)
            kind = "environment"
        except KovaraError:
            config = load_evaluation_config(path)
            if config.comparison is not None:
                load_comparison_environment_configs(config)
            kind = "evaluation"
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "configuration_valid",
        path=str(path),
        kind=kind,
        schema_version=config.schema_version,
    )


@environment_app.command("run")
def run_environment(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    agent: PolicyName = "frontier",
    seed: Annotated[int, typer.Option(min=0)] = 0,
    render: Annotated[str, typer.Option(help="Render mode: ansi or none.")] = "ansi",
) -> None:
    """Run one baseline-controlled environment episode."""

    if render not in {"ansi", "none"}:
        raise typer.BadParameter("render must be 'ansi' or 'none'")
    try:
        env_config = load_environment_config(config)
    except KovaraError as exc:
        _abort(exc)
    policy_type = _policy_factory(agent)
    env = GridRescueParallelEnv(
        env_config,
        render_mode="ansi" if render == "ansi" else None,
    )
    observations, _infos = env.reset(seed=seed)
    policies = {agent_id: policy_type() for agent_id in env.possible_agents}
    for slot, agent_id in enumerate(env.possible_agents):
        policies[agent_id].reset(
            agent_id=agent_id,
            observation_space=env.observation_space(agent_id),
            action_space=env.action_space(agent_id),
            seed=derive_seed(seed, "policy", slot),
        )
    if render == "ansi":
        typer.echo(env.render())
    shared_return = 0.0
    success = False
    while env.agents:
        acting_agents = tuple(env.agents)
        actions = {
            agent_id: policies[agent_id].act(observations[agent_id]) for agent_id in acting_agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        shared_return += next(iter(rewards.values()))
        for agent_id in acting_agents:
            policy_info = PolicyTransitionInfo(
                blocked=bool(infos[agent_id]["blocked"]),
                message_sent=bool(infos[agent_id]["message_sent"]),
                communication_rejected=bool(infos[agent_id]["communication_rejected"]),
            )
            policies[agent_id].observe_outcome(
                reward=rewards[agent_id],
                terminated=terminations[agent_id],
                truncated=truncations[agent_id],
                info=policy_info,
            )
        success = any(terminations.values())
        if render == "ansi":
            typer.echo()
            typer.echo(env.render())
    structlog.get_logger().info(
        "episode_complete",
        environment=env.metadata["name"],
        policy=agent,
        seed=seed,
        success=success,
        steps=env.snapshot.step_count,
        shared_return=shared_return,
    )
    env.close()


@app.command("evaluate")
def evaluate(
    eval_config: Annotated[
        Path,
        typer.Option("--eval-config", exists=True, dir_okay=False),
    ],
    env_config: Annotated[
        Path | None,
        typer.Option("--env-config", exists=True, dir_okay=False),
    ] = None,
    agent: PolicyName = "frontier",
    output: Annotated[Path, typer.Option("--output")] = Path("runs/evaluation"),
) -> None:
    """Evaluate a baseline and persist transparent local artifacts."""

    try:
        evaluation = load_evaluation_config(eval_config)
        held_out_environment = None
        if evaluation.comparison is not None:
            if env_config is not None:
                raise ConfigurationError(
                    "--env-config must be omitted when the evaluation configuration "
                    "declares an authoritative comparison"
                )
            environment, held_out_environment = load_comparison_environment_configs(evaluation)
            environment_label = str(evaluation.comparison.reference_environment)
        else:
            if env_config is None:
                raise ConfigurationError("--env-config is required when no comparison is declared")
            environment = load_environment_config(env_config)
            environment_label = str(env_config.resolve())
        policy_type = _policy_factory(agent)
        logger = structlog.get_logger()
        logger.info(
            "evaluation_started",
            policy=agent,
            episodes=len(evaluation.resolved_seeds),
            environment=environment_label,
        )
        result = evaluate_policy(
            env_config=environment,
            evaluation_config=evaluation,
            policy_factory=policy_type,
        )
        held_out_result = None
        comparison = None
        if held_out_environment is not None:
            held_out_result = evaluate_policy(
                env_config=held_out_environment,
                evaluation_config=evaluation,
                policy_factory=policy_type,
            )
            comparison = comparison_summary(
                result,
                held_out_result,
                environment,
                held_out_environment,
            )
        ArtifactWriter(output).write(
            env_config=environment,
            evaluation_config=evaluation,
            result=result,
            held_out_env_config=held_out_environment,
            held_out_result=held_out_result,
            comparison=comparison,
        )
    except KovaraError as exc:
        _abort(exc)
    logger.info(
        "evaluation_complete",
        output=str(output),
        success_rate=result.summary.metrics["success_rate"].mean,
    )


if __name__ == "__main__":
    app()
