"""Full single-process training loop for the v0.1 MAPPO-style learner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import Field, ValidationError

from kovara9.config.loader import TrainingInputs, configuration_fingerprint
from kovara9.config.models import EnvConfig, StrictModel
from kovara9.core.errors import TrainingError
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.evaluation.runner import evaluate_policy
from kovara9.reporting.training_artifacts import TrainingArtifactStatus, TrainingArtifactWriter
from kovara9.training.checkpoint import (
    CheckpointMetadata,
    CheckpointProgress,
    LearnerSignature,
    LoadedCheckpoint,
    load_training_checkpoint,
    restore_actor_state,
    restore_model_states,
    save_training_checkpoint,
    training_definition_fingerprint,
)
from kovara9.training.collector import SynchronousRolloutCollector
from kovara9.training.config import TrainingConfig
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.evaluation import actor_policy_factory
from kovara9.training.gae import compute_gae
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.optimization import PPOOptimizer, PPOUpdateDiagnostics
from kovara9.training.protocols import TrainingProgress, TrainingResult
from kovara9.training.runtime import configure_deterministic_algorithms, resolve_device
from kovara9.training.seeds import ExperimentSeedStreams


class TrainingUpdateRecord(StrictModel):
    """One finite optimizer update and any scheduled validation result."""

    environment_steps: int = Field(gt=0)
    optimizer_updates: int = Field(gt=0)
    completed_episodes: int = Field(ge=0)
    rollout_completed_episodes: int = Field(ge=0)
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    move_entropy: float
    message_entropy: float
    approximate_kl: float
    clip_fraction: float
    mean_probability_ratio: float
    explained_variance: float | None
    maximum_pre_clip_gradient_norm: float = Field(ge=0.0)
    maximum_post_clip_gradient_norm: float = Field(ge=0.0)
    valid_sample_count: int = Field(gt=0)
    minibatch_count: int = Field(gt=0)
    validation_metrics: dict[str, float] | None


@dataclass(slots=True)
class _LearnerComponents:
    actor: SharedActor
    critic: CentralizedCritic
    collector: SynchronousRolloutCollector
    optimizer: PPOOptimizer
    signature: LearnerSignature


def probe_learner_signature(environment: EnvConfig) -> LearnerSignature:
    """Derive observation, state, action, and homogeneous-agent contracts."""

    probe = GridRescueParallelEnv(environment)
    try:
        agent_order = tuple(probe.possible_agents)
        actor_encoder = ActorObservationEncoder(probe.observation_space(agent_order[0]))
        critic_encoder = CentralStateEncoder(probe.state_space)
    finally:
        probe.close()
    return LearnerSignature(
        actor_input_dim=actor_encoder.input_dim,
        critic_input_dim=critic_encoder.input_dim,
        move_action_count=actor_encoder.move_action_count,
        message_action_count=actor_encoder.message_action_count,
        agent_order=agent_order,
    )


def actor_from_checkpoint(
    checkpoint: LoadedCheckpoint,
    *,
    environment: EnvConfig,
    device: torch.device,
) -> SharedActor:
    """Build a deterministic inference actor after local-signature validation."""

    actual = probe_learner_signature(environment)
    expected = checkpoint.metadata.signature
    if (
        actual.actor_input_dim != expected.actor_input_dim
        or actual.move_action_count != expected.move_action_count
        or actual.message_action_count != expected.message_action_count
    ):
        raise TrainingError(
            "checkpoint actor is incompatible with the environment observation/action signature"
        )
    streams = ExperimentSeedStreams(checkpoint.metadata.training_config.seed)
    actor = SharedActor(
        input_dim=expected.actor_input_dim,
        move_action_count=expected.move_action_count,
        message_action_count=expected.message_action_count,
        config=checkpoint.metadata.training_config.network,
        seed=streams.actor_initialization,
    ).to(device)
    restore_actor_state(checkpoint, actor=actor)
    actor.eval()
    return actor


def untrained_actor_from_checkpoint_definition(
    checkpoint: LoadedCheckpoint,
    *,
    environment: EnvConfig,
    device: torch.device,
) -> SharedActor:
    """Recreate the declared initialization for a factual untrained comparison."""

    actual = probe_learner_signature(environment)
    expected = checkpoint.metadata.signature
    if (
        actual.actor_input_dim != expected.actor_input_dim
        or actual.move_action_count != expected.move_action_count
        or actual.message_action_count != expected.message_action_count
    ):
        raise TrainingError(
            "checkpoint actor definition is incompatible with the environment signature"
        )
    streams = ExperimentSeedStreams(checkpoint.metadata.training_config.seed)
    return SharedActor(
        input_dim=expected.actor_input_dim,
        move_action_count=expected.move_action_count,
        message_action_count=expected.message_action_count,
        config=checkpoint.metadata.training_config.network,
        seed=streams.actor_initialization,
    ).to(device)


class MAPPOTrainer:
    """Synchronous trainer for the single approved v0.1 learning algorithm."""

    def __init__(self, inputs: TrainingInputs, *, project_root: Path | None = None) -> None:
        self.inputs = TrainingInputs(
            training=TrainingConfig.model_validate(
                inputs.training.model_dump(mode="python", round_trip=True)
            ),
            environment=EnvConfig.model_validate(
                inputs.environment.model_dump(mode="python", round_trip=True)
            ),
            validation=type(inputs.validation).model_validate(
                inputs.validation.model_dump(mode="python", round_trip=True)
            ),
        )
        self.project_root = project_root

    @property
    def name(self) -> str:
        return self.inputs.training.algorithm

    def train(
        self,
        *,
        output_dir: Path,
        resume_from: Path | None = None,
        stop_after_environment_steps: int | None = None,
    ) -> TrainingResult:
        """Train to the configured total or an explicit rollout-aligned boundary."""

        config = self.inputs.training
        device = resolve_device(config.device)
        configure_deterministic_algorithms(config.deterministic_torch)
        components = self._build_components(device)
        try:
            checkpoint = load_training_checkpoint(resume_from) if resume_from is not None else None
            records: list[TrainingUpdateRecord] = []
            progress = TrainingProgress(0, 0, 0)
            if checkpoint is not None:
                progress, records = self._restore_checkpoint(checkpoint, components)
            target = self._resolve_target(progress, stop_after_environment_steps)
            writer = TrainingArtifactWriter(output_dir, project_root=self.project_root)
            writer.start(self.inputs, resume_from=resume_from)
            latest_checkpoint: Path | None = None
            while progress.environment_steps < target:
                collection = components.collector.collect(deterministic=False)
                gae = compute_gae(
                    collection.batch,
                    gamma=config.discount_factor,
                    gae_lambda=config.gae_lambda,
                    normalize_advantages=config.normalize_advantages,
                    normalization_epsilon=config.advantage_normalization_epsilon,
                )
                update = components.optimizer.update(collection.batch, gae)
                progress = TrainingProgress(
                    environment_steps=(
                        progress.environment_steps + config.rollout_environment_steps
                    ),
                    optimizer_updates=progress.optimizer_updates + 1,
                    completed_episodes=(
                        progress.completed_episodes + len(collection.completed_episodes)
                    ),
                )
                validation_metrics = (
                    self._evaluate_validation(components.actor, device, progress)
                    if progress.environment_steps % config.evaluation_frequency == 0
                    else None
                )
                records.append(
                    self._training_record(
                        progress,
                        update,
                        rollout_completed_episodes=len(collection.completed_episodes),
                        validation_metrics=validation_metrics,
                    )
                )
                serialized_records = tuple(record.model_dump(mode="json") for record in records)
                writer.write_metrics(serialized_records)
                checkpoint_due = (
                    progress.environment_steps % config.checkpoint_frequency == 0
                    or progress.environment_steps == target
                )
                if checkpoint_due:
                    latest_checkpoint = writer.checkpoint_path(progress.environment_steps)
                    complete = progress.environment_steps == config.total_environment_steps
                    self._save_checkpoint(
                        latest_checkpoint,
                        components=components,
                        progress=progress,
                        records=serialized_records,
                        training_complete=complete,
                    )
                    status: TrainingArtifactStatus = (
                        "complete"
                        if complete
                        else "bounded"
                        if progress.environment_steps == target
                        else "running"
                    )
                    writer.update_manifest(
                        self.inputs,
                        status=status,
                        progress=progress,
                        latest_checkpoint=latest_checkpoint,
                    )
            if latest_checkpoint is None:
                raise TrainingError("training ended without publishing a checkpoint")
            return TrainingResult(checkpoint=latest_checkpoint, progress=progress)
        finally:
            components.collector.close()

    def _build_components(self, device: torch.device) -> _LearnerComponents:
        signature = probe_learner_signature(self.inputs.environment)
        streams = ExperimentSeedStreams(self.inputs.training.seed)
        actor = SharedActor(
            input_dim=signature.actor_input_dim,
            move_action_count=signature.move_action_count,
            message_action_count=signature.message_action_count,
            config=self.inputs.training.network,
            seed=streams.actor_initialization,
        ).to(device)
        critic = CentralizedCritic(
            input_dim=signature.critic_input_dim,
            config=self.inputs.training.network,
            seed=streams.critic_initialization,
        ).to(device)
        collector = SynchronousRolloutCollector(
            environment_factory=lambda: GridRescueParallelEnv(self.inputs.environment),
            num_environments=self.inputs.training.num_environments,
            rollout_length=self.inputs.training.rollout_length,
            actor=actor,
            critic=critic,
            root_seed=self.inputs.training.seed,
            device=device,
        )
        optimizer = PPOOptimizer(
            actor=actor,
            critic=critic,
            config=self.inputs.training,
            shuffle_seed=streams.optimizer_shuffle,
        )
        return _LearnerComponents(actor, critic, collector, optimizer, signature)

    def _restore_checkpoint(
        self,
        checkpoint: LoadedCheckpoint,
        components: _LearnerComponents,
    ) -> tuple[TrainingProgress, list[TrainingUpdateRecord]]:
        metadata = checkpoint.metadata
        expected_fingerprints = (
            training_definition_fingerprint(self.inputs.training),
            configuration_fingerprint(self.inputs.environment),
            configuration_fingerprint(self.inputs.validation),
        )
        actual_fingerprints = (
            metadata.training_fingerprint,
            metadata.environment_fingerprint,
            metadata.validation_fingerprint,
        )
        if (
            training_definition_fingerprint(metadata.training_config)
            != metadata.training_fingerprint
        ):
            raise TrainingError("checkpoint training configuration fingerprint is inconsistent")
        if actual_fingerprints != expected_fingerprints:
            raise TrainingError("checkpoint configuration fingerprints do not match resume inputs")
        if metadata.signature != components.signature:
            raise TrainingError("checkpoint learner signature does not match resume environment")
        progress = metadata.progress.to_progress()
        rollout_steps = self.inputs.training.rollout_environment_steps
        if progress.environment_steps % rollout_steps != 0:
            raise TrainingError("checkpoint progress is not aligned to a complete rollout")
        if progress.optimizer_updates != progress.environment_steps // rollout_steps:
            raise TrainingError("checkpoint optimizer counter does not match environment steps")
        if progress.environment_steps > self.inputs.training.total_environment_steps:
            raise TrainingError("checkpoint progress exceeds configured training total")
        if metadata.training_complete != (
            progress.environment_steps == self.inputs.training.total_environment_steps
        ):
            raise TrainingError("checkpoint completion flag is inconsistent with progress")
        try:
            records = [
                TrainingUpdateRecord.model_validate(record)
                for record in checkpoint.training_records
            ]
        except ValidationError as exc:
            raise TrainingError(f"checkpoint training metric history is invalid: {exc}") from exc
        if len(records) != progress.optimizer_updates:
            raise TrainingError("checkpoint metric history does not match optimizer counter")
        if records and records[-1].environment_steps != progress.environment_steps:
            raise TrainingError("checkpoint metric history does not match progress")

        restore_model_states(checkpoint, actor=components.actor, critic=components.critic)
        components.optimizer.restore_checkpoint_state(checkpoint.optimizer_state)
        components.collector.restore_checkpoint_state(checkpoint.collector_state)
        return progress, records

    def _resolve_target(
        self,
        progress: TrainingProgress,
        requested: int | None,
    ) -> int:
        target = self.inputs.training.total_environment_steps if requested is None else requested
        if target <= progress.environment_steps:
            raise TrainingError(
                "training target must be greater than checkpoint/current environment steps"
            )
        if target > self.inputs.training.total_environment_steps:
            raise TrainingError("training target cannot exceed configured total_environment_steps")
        if target % self.inputs.training.rollout_environment_steps != 0:
            raise TrainingError(
                "training target must be divisible by rollout_length * num_environments"
            )
        return target

    def _evaluate_validation(
        self,
        actor: SharedActor,
        device: torch.device,
        progress: TrainingProgress,
    ) -> dict[str, float]:
        factory = actor_policy_factory(
            actor=actor,
            device=device,
            policy_name="shared-actor-validation",
            parameters={
                "deterministic": True,
                "training_seed": self.inputs.training.seed,
                "optimizer_updates": progress.optimizer_updates,
            },
        )
        result = evaluate_policy(
            env_config=self.inputs.environment,
            evaluation_config=self.inputs.validation,
            policy_factory=factory,
        )
        return {name: summary.mean for name, summary in result.summary.metrics.items()}

    @staticmethod
    def _training_record(
        progress: TrainingProgress,
        update: PPOUpdateDiagnostics,
        *,
        rollout_completed_episodes: int,
        validation_metrics: dict[str, float] | None,
    ) -> TrainingUpdateRecord:
        return TrainingUpdateRecord(
            environment_steps=progress.environment_steps,
            optimizer_updates=progress.optimizer_updates,
            completed_episodes=progress.completed_episodes,
            rollout_completed_episodes=rollout_completed_episodes,
            total_loss=update.total_loss,
            policy_loss=update.policy_loss,
            value_loss=update.value_loss,
            entropy=update.entropy,
            move_entropy=update.move_entropy,
            message_entropy=update.message_entropy,
            approximate_kl=update.approximate_kl,
            clip_fraction=update.clip_fraction,
            mean_probability_ratio=update.mean_probability_ratio,
            explained_variance=update.explained_variance,
            maximum_pre_clip_gradient_norm=update.maximum_pre_clip_gradient_norm,
            maximum_post_clip_gradient_norm=update.maximum_post_clip_gradient_norm,
            valid_sample_count=update.valid_sample_count,
            minibatch_count=update.minibatch_count,
            validation_metrics=validation_metrics,
        )

    def _save_checkpoint(
        self,
        path: Path,
        *,
        components: _LearnerComponents,
        progress: TrainingProgress,
        records: tuple[dict[str, object], ...],
        training_complete: bool,
    ) -> None:
        metadata = CheckpointMetadata(
            training_config=self.inputs.training,
            training_fingerprint=training_definition_fingerprint(self.inputs.training),
            environment_fingerprint=configuration_fingerprint(self.inputs.environment),
            validation_fingerprint=configuration_fingerprint(self.inputs.validation),
            signature=components.signature,
            progress=CheckpointProgress.from_progress(progress),
            training_complete=training_complete,
        )
        save_training_checkpoint(
            path,
            metadata=metadata,
            actor=components.actor,
            critic=components.critic,
            optimizer_state=components.optimizer.checkpoint_state(),
            collector_state=components.collector.checkpoint_state(),
            training_records=records,
        )
