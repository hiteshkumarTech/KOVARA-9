"""Command-line entry point for simulation and evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import structlog
import torch
import typer

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.policy import Policy, PolicyTransitionInfo
from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import (
    TrainingInputs,
    load_comparison_environment_configs,
    load_environment_config,
    load_evaluation_config,
    load_training_inputs,
)
from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import ConfigurationError, KovaraError, NumericalError, TrainingError
from kovara9.core.seeding import derive_seed
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.evaluation.runner import evaluate_policy
from kovara9.reporting.artifacts import ArtifactWriter
from kovara9.reporting.summaries import comparison_summary
from kovara9.training.collector import SynchronousRolloutCollector
from kovara9.training.config import TrainingConfig
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.gae import compute_gae
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.optimization import PPOOptimizer
from kovara9.training.runtime import configure_deterministic_algorithms, resolve_device
from kovara9.training.seeds import ExperimentSeedStreams

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


def _make_rollout_collector(
    inputs: TrainingInputs,
    *,
    steps: int,
    device: torch.device,
) -> SynchronousRolloutCollector:
    probe = GridRescueParallelEnv(inputs.environment)
    try:
        actor_encoder = ActorObservationEncoder(probe.observation_space(probe.possible_agents[0]))
        critic_encoder = CentralStateEncoder(probe.state_space)
    finally:
        probe.close()
    streams = ExperimentSeedStreams(inputs.training.seed)
    actor = SharedActor(
        input_dim=actor_encoder.input_dim,
        move_action_count=actor_encoder.move_action_count,
        message_action_count=actor_encoder.message_action_count,
        config=inputs.training.network,
        seed=streams.actor_initialization,
    )
    critic = CentralizedCritic(
        input_dim=critic_encoder.input_dim,
        config=inputs.training.network,
        seed=streams.critic_initialization,
    )
    return SynchronousRolloutCollector(
        environment_factory=lambda: GridRescueParallelEnv(inputs.environment),
        num_environments=inputs.training.num_environments,
        rollout_length=steps,
        actor=actor,
        critic=critic,
        root_seed=inputs.training.seed,
        device=device,
    )


@config_app.command("validate")
def validate_config(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate an environment or evaluation YAML configuration."""

    config: EnvConfig | EvaluationConfig | TrainingConfig
    try:
        try:
            config = load_environment_config(path)
            kind = "environment"
        except KovaraError:
            try:
                evaluation = load_evaluation_config(path)
            except KovaraError:
                config = load_training_inputs(path).training
                kind = "training"
            else:
                config = evaluation
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


@app.command("rollout-smoke")
def rollout_smoke(
    training_config: Annotated[
        Path,
        typer.Option("--training-config", exists=True, dir_okay=False, readable=True),
    ],
    steps: Annotated[int | None, typer.Option("--steps", min=1)] = None,
) -> None:
    """Collect a bounded untrained rollout; do not optimize or save a model."""

    collector: SynchronousRolloutCollector | None = None
    try:
        inputs = load_training_inputs(training_config)
        rollout_steps = inputs.training.rollout_length if steps is None else steps
        device = resolve_device(inputs.training.device)
        configure_deterministic_algorithms(inputs.training.deterministic_torch)
        collector = _make_rollout_collector(inputs, steps=rollout_steps, device=device)
        collection = collector.collect(deterministic=False)
    except KovaraError as exc:
        _abort(exc)
    finally:
        if collector is not None:
            collector.close()
    batch = collection.batch
    structlog.get_logger().info(
        "rollout_smoke_complete",
        benchmark=False,
        training_performed=False,
        seed=collection.root_seed,
        device=str(device),
        rollout_steps=rollout_steps,
        num_environments=inputs.training.num_environments,
        agent_order=list(batch.agent_order),
        actor_shape=list(batch.actor_features.shape),
        critic_shape=list(batch.critic_features.shape),
        transition_count=(
            rollout_steps * inputs.training.num_environments * len(batch.agent_order)
        ),
        completed_episodes=len(collection.completed_episodes),
        reset_seeds=[list(seeds) for seeds in collection.reset_seeds],
    )


@app.command("update-smoke")
def update_smoke(
    training_config: Annotated[
        Path,
        typer.Option("--training-config", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Run one bounded optimization smoke; do not save or claim a learned policy."""

    collector: SynchronousRolloutCollector | None = None
    try:
        inputs = load_training_inputs(training_config)
        device = resolve_device(inputs.training.device)
        configure_deterministic_algorithms(inputs.training.deterministic_torch)
        collector = _make_rollout_collector(
            inputs,
            steps=inputs.training.rollout_length,
            device=device,
        )
        collection = collector.collect(deterministic=False)
        gae = compute_gae(
            collection.batch,
            gamma=inputs.training.discount_factor,
            gae_lambda=inputs.training.gae_lambda,
            normalize_advantages=inputs.training.normalize_advantages,
            normalization_epsilon=inputs.training.advantage_normalization_epsilon,
        )
        actor_before = tuple(
            parameter.detach().clone() for parameter in collector.actor.parameters()
        )
        critic_before = tuple(
            parameter.detach().clone() for parameter in collector.critic.parameters()
        )
        streams = ExperimentSeedStreams(inputs.training.seed)
        update = PPOOptimizer(
            actor=collector.actor,
            critic=collector.critic,
            config=inputs.training,
            shuffle_seed=streams.optimizer_shuffle,
        ).update(collection.batch, gae)
        actor_changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                actor_before,
                collector.actor.parameters(),
                strict=True,
            )
        )
        critic_changed = any(
            not torch.equal(before, after)
            for before, after in zip(
                critic_before,
                collector.critic.parameters(),
                strict=True,
            )
        )
        parameters_finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in (*collector.actor.parameters(), *collector.critic.parameters())
        )
        if not actor_changed or not critic_changed:
            raise TrainingError(
                "optimization smoke did not change both actor and critic parameters"
            )
        if not parameters_finite:
            raise NumericalError("optimization smoke produced non-finite model parameters")
    except KovaraError as exc:
        _abort(exc)
    finally:
        if collector is not None:
            collector.close()
    structlog.get_logger().info(
        "optimization_smoke_complete",
        optimization_smoke_test=True,
        benchmark=False,
        useful_policy_learned=False,
        full_training_run=False,
        checkpoint_saved=False,
        seed=inputs.training.seed,
        optimizer_shuffle_seed=streams.optimizer_shuffle,
        device=str(device),
        rollout_steps=inputs.training.rollout_length,
        valid_sample_count=update.valid_sample_count,
        minibatch_count=update.minibatch_count,
        actor_parameters_changed=actor_changed,
        critic_parameters_changed=critic_changed,
        parameters_finite=parameters_finite,
        total_loss=update.total_loss,
        policy_loss=update.policy_loss,
        value_loss=update.value_loss,
        entropy=update.entropy,
        approximate_kl=update.approximate_kl,
        clip_fraction=update.clip_fraction,
        maximum_pre_clip_gradient_norm=update.maximum_pre_clip_gradient_norm,
        maximum_post_clip_gradient_norm=update.maximum_post_clip_gradient_norm,
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
