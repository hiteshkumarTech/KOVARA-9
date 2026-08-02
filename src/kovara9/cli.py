"""Command-line entry point for simulation and evaluation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, cast

import structlog
import torch
import typer

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.policy import Policy, PolicyTransitionInfo
from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import (
    TrainingInputs,
    configuration_fingerprint,
    load_comparison_environment_configs,
    load_environment_config,
    load_evaluation_config,
    load_training_inputs,
)
from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import (
    ArtifactError,
    ConfigurationError,
    KovaraError,
    NumericalError,
    TrainingError,
)
from kovara9.core.seeding import derive_seed
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.evaluation.runner import EvaluationResult, PolicyFactory, evaluate_policy
from kovara9.experiments.day6 import (
    load_candidate_freeze,
    load_day6_training_inputs,
    validate_candidate_freeze,
)
from kovara9.experiments.day8 import reject_consumed_test_partition, run_final_evaluation
from kovara9.reporting.artifacts import ArtifactWriter
from kovara9.reporting.summaries import comparison_summary
from kovara9.training.checkpoint import (
    checkpoint_sha256,
    load_training_checkpoint,
    model_state_sha256,
)
from kovara9.training.collector import SynchronousRolloutCollector
from kovara9.training.config import DeviceName, TrainingConfig
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.evaluation import actor_policy_factory
from kovara9.training.gae import compute_gae
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.optimization import PPOOptimizer
from kovara9.training.runtime import configure_deterministic_algorithms, resolve_device
from kovara9.training.seeds import ExperimentSeedStreams
from kovara9.training.trainer import (
    MAPPOTrainer,
    actor_from_checkpoint,
    untrained_actor_from_checkpoint_definition,
)

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


def _resolve_requested_device(name: str) -> torch.device:
    if name not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter("device must be 'auto', 'cpu', or 'cuda'")
    return resolve_device(cast(DeviceName, name))


def _evaluation_environments(
    evaluation: EvaluationConfig,
    env_config: Path | None,
) -> tuple[EnvConfig, EnvConfig | None, str]:
    if evaluation.comparison is not None:
        if env_config is not None:
            raise ConfigurationError(
                "--env-config must be omitted when the evaluation configuration "
                "declares an authoritative comparison"
            )
        environment, held_out = load_comparison_environment_configs(evaluation)
        return environment, held_out, str(evaluation.comparison.reference_environment)
    if env_config is None:
        raise ConfigurationError("--env-config is required when no comparison is declared")
    return load_environment_config(env_config), None, str(env_config.resolve())


def _evaluate_and_write(
    *,
    evaluation: EvaluationConfig,
    environment: EnvConfig,
    held_out_environment: EnvConfig | None,
    policy_factory: PolicyFactory,
    output: Path,
) -> EvaluationResult:
    result = evaluate_policy(
        env_config=environment,
        evaluation_config=evaluation,
        policy_factory=policy_factory,
    )
    held_out_result = None
    comparison = None
    if held_out_environment is not None:
        held_out_result = evaluate_policy(
            env_config=held_out_environment,
            evaluation_config=evaluation,
            policy_factory=policy_factory,
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
    return result


def _atomic_cli_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot write policy comparison artifact {path}: {exc}") from exc


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


@config_app.command("verify-candidate")
def verify_candidate(
    candidate: Annotated[
        Path,
        typer.Option("--candidate", exists=True, dir_okay=False, readable=True),
    ],
    freeze_record: Annotated[
        Path,
        typer.Option("--freeze-record", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Verify that a frozen candidate and every bound experiment identity still match."""

    try:
        freeze = load_candidate_freeze(freeze_record)
        validate_candidate_freeze(candidate, freeze)
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "candidate_freeze_verified",
        candidate=str(candidate),
        configuration_fingerprint=freeze.configuration_fingerprint,
        test_partition_consumed=Path("configs/evaluation/final_test_consumed.json").is_file(),
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


@app.command("train")
def train(
    training_config: Annotated[
        Path,
        typer.Option("--training-config", exists=True, dir_okay=False, readable=True),
    ],
    output: Annotated[Path, typer.Option("--output")] = Path("runs/training"),
    resume_from: Annotated[
        Path | None,
        typer.Option("--resume-from", exists=True, dir_okay=False, readable=True),
    ] = None,
    stop_after_environment_steps: Annotated[
        int | None,
        typer.Option(
            "--stop-after-environment-steps",
            min=1,
            help="Stop at an aligned checkpoint boundary for interruption/resume checks.",
        ),
    ] = None,
    initialize_only: Annotated[
        bool,
        typer.Option(
            "--initialize-only",
            help="Save the exact untrained state without collecting or optimizing.",
        ),
    ] = False,
) -> None:
    """Run configured training and publish atomic checkpoints and metrics."""

    try:
        inputs = load_training_inputs(training_config)
        trainer = MAPPOTrainer(inputs)
        if initialize_only:
            if resume_from is not None or stop_after_environment_steps is not None:
                raise TrainingError(
                    "--initialize-only cannot be combined with resume or a training boundary"
                )
            result = trainer.initialize(output_dir=output)
        else:
            result = trainer.train(
                output_dir=output,
                resume_from=resume_from,
                stop_after_environment_steps=stop_after_environment_steps,
            )
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "training_initialized"
        if initialize_only
        else "training_complete"
        if result.progress.environment_steps == inputs.training.total_environment_steps
        else "training_bounded",
        benchmark=False,
        useful_policy_learned=False,
        output=str(output),
        checkpoint=str(result.checkpoint),
        environment_steps=result.progress.environment_steps,
        optimizer_updates=result.progress.optimizer_updates,
        completed_episodes=result.progress.completed_episodes,
    )


@app.command("day6-run-seed")
def day6_run_seed(
    training_config: Annotated[
        Path,
        typer.Option("--training-config", exists=True, dir_okay=False, readable=True),
    ],
    root_seed: Annotated[int, typer.Option("--root-seed", min=0)],
    output: Annotated[Path, typer.Option("--output")],
    initialize_only: Annotated[
        bool,
        typer.Option(
            "--initialize-only",
            help="Save the exact Day 6 initialization before its training run.",
        ),
    ] = False,
) -> None:
    """Run one controlled Day 6 root seed using validation evidence only."""

    try:
        inputs = load_day6_training_inputs(training_config, root_seed=root_seed)
        trainer = MAPPOTrainer(inputs)
        result = (
            trainer.initialize(output_dir=output)
            if initialize_only
            else trainer.train(output_dir=output)
        )
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "day6_seed_initialized" if initialize_only else "day6_seed_training_complete",
        benchmark=False,
        test_partition_consumed=False,
        root_seed=root_seed,
        output=str(output),
        checkpoint=str(result.checkpoint),
        environment_steps=result.progress.environment_steps,
        optimizer_updates=result.progress.optimizer_updates,
        completed_episodes=result.progress.completed_episodes,
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
        reject_consumed_test_partition(evaluation)
        environment, held_out_environment, environment_label = _evaluation_environments(
            evaluation, env_config
        )
        policy_type = _policy_factory(agent)
        logger = structlog.get_logger()
        logger.info(
            "evaluation_started",
            policy=agent,
            episodes=len(evaluation.resolved_seeds),
            environment=environment_label,
        )
        result = _evaluate_and_write(
            evaluation=evaluation,
            environment=environment,
            held_out_environment=held_out_environment,
            policy_factory=policy_type,
            output=output,
        )
    except KovaraError as exc:
        _abort(exc)
    logger.info(
        "evaluation_complete",
        output=str(output),
        success_rate=result.summary.metrics["success_rate"].mean,
    )


@app.command("evaluate-checkpoint")
def evaluate_checkpoint(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True),
    ],
    eval_config: Annotated[
        Path,
        typer.Option("--eval-config", exists=True, dir_okay=False, readable=True),
    ],
    env_config: Annotated[
        Path | None,
        typer.Option("--env-config", exists=True, dir_okay=False, readable=True),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("runs/checkpoint-evaluation"),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Evaluate a saved actor deterministically without constructing a critic."""

    try:
        loaded = load_training_checkpoint(checkpoint)
        evaluation = load_evaluation_config(eval_config)
        reject_consumed_test_partition(evaluation)
        environment, held_out_environment, environment_label = _evaluation_environments(
            evaluation, env_config
        )
        resolved_device = _resolve_requested_device(device)
        configure_deterministic_algorithms(loaded.metadata.training_config.deterministic_torch)
        actor = actor_from_checkpoint(
            loaded,
            environment=environment,
            device=resolved_device,
        )
        policy_factory = actor_policy_factory(
            actor=actor,
            device=resolved_device,
            policy_name="checkpoint-shared-actor",
            parameters={
                "checkpoint_sha256": checkpoint_sha256(checkpoint),
                "actor_state_sha256": model_state_sha256(loaded.actor_state),
                "deterministic": True,
                "environment_steps": loaded.metadata.progress.environment_steps,
                "training_seed": loaded.metadata.training_config.seed,
            },
        )
        structlog.get_logger().info(
            "checkpoint_evaluation_started",
            checkpoint=str(checkpoint),
            environment=environment_label,
            device=str(resolved_device),
        )
        result = _evaluate_and_write(
            evaluation=evaluation,
            environment=environment,
            held_out_environment=held_out_environment,
            policy_factory=policy_factory,
            output=output,
        )
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "checkpoint_evaluation_complete",
        output=str(output),
        success_rate=result.summary.metrics["success_rate"].mean,
    )


@app.command("compare-policies")
def compare_policies(  # noqa: PLR0913, PLR0917
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", exists=True, dir_okay=False, readable=True),
    ],
    eval_config: Annotated[
        Path,
        typer.Option("--eval-config", exists=True, dir_okay=False, readable=True),
    ],
    env_config: Annotated[
        Path | None,
        typer.Option("--env-config", exists=True, dir_okay=False, readable=True),
    ] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("runs/policy-comparison"),
    device: Annotated[str, typer.Option("--device")] = "cpu",
    allow_test_partition: Annotated[
        bool,
        typer.Option(
            "--allow-test-partition",
            help="Explicitly authorize a frozen test evaluation; never use while tuning.",
        ),
    ] = False,
) -> None:
    """Compare random, frontier, untrained, and checkpoint actors on paired seeds."""

    try:
        loaded = load_training_checkpoint(checkpoint)
        evaluation = load_evaluation_config(eval_config)
        reject_consumed_test_partition(evaluation)
        if evaluation.seed_partition == "test" and not allow_test_partition:
            raise ConfigurationError(
                "policy comparison refuses test seeds during tuning; "
                "use --allow-test-partition only after configuration freeze"
            )
        environment, held_out_environment, _label = _evaluation_environments(evaluation, env_config)
        environment_identity = configuration_fingerprint(environment)
        if environment_identity != loaded.metadata.environment_fingerprint:
            raise ConfigurationError(
                "checkpoint and policy-comparison environment fingerprints differ"
            )
        resolved_device = _resolve_requested_device(device)
        configure_deterministic_algorithms(loaded.metadata.training_config.deterministic_torch)
        trained_actor = actor_from_checkpoint(
            loaded,
            environment=environment,
            device=resolved_device,
        )
        untrained_actor = untrained_actor_from_checkpoint_definition(
            loaded,
            environment=environment,
            device=resolved_device,
        )
        checkpoint_identity = checkpoint_sha256(checkpoint)
        trained_actor_identity = model_state_sha256(loaded.actor_state)
        untrained_actor_identity = model_state_sha256(untrained_actor.state_dict())
        factories: dict[str, PolicyFactory] = {
            "random": RandomPolicy,
            "frontier": FrontierPolicy,
            "untrained": actor_policy_factory(
                actor=untrained_actor,
                device=resolved_device,
                policy_name="untrained-shared-actor",
                parameters={
                    "actor_state_sha256": untrained_actor_identity,
                    "deterministic": True,
                    "training_seed": loaded.metadata.training_config.seed,
                },
            ),
            "checkpoint": actor_policy_factory(
                actor=trained_actor,
                device=resolved_device,
                policy_name="checkpoint-shared-actor",
                parameters={
                    "actor_state_sha256": trained_actor_identity,
                    "checkpoint_sha256": checkpoint_identity,
                    "deterministic": True,
                    "environment_steps": loaded.metadata.progress.environment_steps,
                    "training_seed": loaded.metadata.training_config.seed,
                },
            ),
        }
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactError(f"output directory already exists: {output}") from exc
        except OSError as exc:
            raise ArtifactError(
                f"cannot create policy comparison directory {output}: {exc}"
            ) from exc
        summaries: dict[str, dict[str, float]] = {}
        results: dict[str, EvaluationResult] = {}
        for name, policy_factory in factories.items():
            result = _evaluate_and_write(
                evaluation=evaluation,
                environment=environment,
                held_out_environment=held_out_environment,
                policy_factory=policy_factory,
                output=output / name,
            )
            summaries[name] = {
                metric: summary.mean for metric, summary in result.summary.metrics.items()
            }
            results[name] = result
        paired_records = [
            {
                "seed": seed,
                "policies": {
                    name: results[name].records[index].to_dict() for name in sorted(results)
                },
            }
            for index, seed in enumerate(evaluation.resolved_seeds)
        ]
        _atomic_cli_json(
            output / "policy-comparison.json",
            {
                "schema_version": 2,
                "status": "complete",
                "checkpoint_sha256": checkpoint_identity,
                "configuration_fingerprints": {
                    "environment": environment_identity,
                    "evaluation": configuration_fingerprint(evaluation),
                    "training": loaded.metadata.training_fingerprint,
                },
                "paired_episode_seeds": list(evaluation.resolved_seeds),
                "policies": summaries,
                "policy_parameters": {
                    name: dict(result.policy_parameters) for name, result in results.items()
                },
                "inference_performance": {
                    name: result.inference_performance.to_dict() for name, result in results.items()
                },
                "paired_results": paired_records,
            },
        )
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "policy_comparison_complete",
        output=str(output),
        policies=sorted(summaries),
    )


@app.command("final-evaluate")
def final_evaluate(  # noqa: PLR0913, PLR0917
    candidate: Annotated[
        Path, typer.Option("--candidate", exists=True, dir_okay=False, readable=True)
    ],
    freeze_record: Annotated[
        Path, typer.Option("--freeze-record", exists=True, dir_okay=False, readable=True)
    ],
    eval_config: Annotated[
        Path, typer.Option("--eval-config", exists=True, dir_okay=False, readable=True)
    ],
    preregistration: Annotated[
        Path, typer.Option("--preregistration", exists=True, dir_okay=False, readable=True)
    ],
    preregistration_sha256: Annotated[str, typer.Option("--preregistration-sha256")],
    artifact_root: Annotated[
        Path, typer.Option("--artifact-root", exists=True, file_okay=False, readable=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    consumed_record: Annotated[Path, typer.Option("--consumed-record")] = Path(
        "configs/evaluation/final_test_consumed.json"
    ),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Run the sole preregistered final held-out evaluation and consume its partition."""

    try:
        if device != "cpu":
            raise ConfigurationError("the preregistered final evaluation requires device=cpu")
        record = run_final_evaluation(
            candidate_path=candidate,
            freeze_path=freeze_record,
            evaluation_path=eval_config,
            preregistration_path=preregistration,
            preregistration_sha256=preregistration_sha256,
            artifact_root=artifact_root,
            output=output,
            consumed_record=consumed_record,
            device_name=device,
        )
    except KovaraError as exc:
        _abort(exc)
    structlog.get_logger().info(
        "final_evaluation_complete",
        output=str(output),
        policies=record["completed_policies"],
        runtime_seconds=record["runtime_seconds"],
    )


if __name__ == "__main__":
    app()
