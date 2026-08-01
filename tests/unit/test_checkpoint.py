from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from kovara9.config.loader import TrainingInputs, configuration_fingerprint, load_training_inputs
from kovara9.config.models import CommunicationConfig
from kovara9.core.errors import TrainingError
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.environments.grid_rescue.state import WorldStateCheckpoint
from kovara9.training.checkpoint import (
    CheckpointMetadata,
    CheckpointProgress,
    LoadedCheckpoint,
    checkpoint_sha256,
    load_training_checkpoint,
    model_state_sha256,
    training_definition_fingerprint,
)
from kovara9.training.optimization import PPOUpdateDiagnostics
from kovara9.training.protocols import TrainingProgress
from kovara9.training.trainer import (
    MAPPOTrainer,
    actor_from_checkpoint,
    probe_learner_signature,
    validation_selection_key,
)


def _smoke_inputs() -> TrainingInputs:
    loaded = load_training_inputs(Path("configs/training/mappo_smoke.yaml"))
    environment = loaded.environment.model_copy(update={"max_steps": 4})
    validation = loaded.validation.model_copy(update={"seeds": (10_000,), "bootstrap_samples": 0})
    training = loaded.training.model_copy(
        update={
            "rollout_length": 4,
            "ppo_epochs": 1,
            "minibatch_size": 8,
            "total_environment_steps": 16,
            "checkpoint_frequency": 8,
            "evaluation_frequency": 16,
        }
    )
    return TrainingInputs(training, environment, validation)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _assert_complete_checkpoints_equal(
    resumed: LoadedCheckpoint,
    uninterrupted: LoadedCheckpoint,
) -> None:
    assert resumed.metadata.progress == uninterrupted.metadata.progress
    assert resumed.training_records == uninterrupted.training_records
    _assert_nested_equal(resumed.actor_state, uninterrupted.actor_state)
    _assert_nested_equal(resumed.critic_state, uninterrupted.critic_state)
    _assert_nested_equal(resumed.optimizer_state, uninterrupted.optimizer_state)
    _assert_nested_equal(resumed.collector_state, uninterrupted.collector_state)


def test_interrupted_resume_exactly_matches_uninterrupted_training(tmp_path: Path) -> None:
    inputs = _smoke_inputs()
    partial = MAPPOTrainer(inputs).train(
        output_dir=tmp_path / "partial",
        stop_after_environment_steps=8,
    )
    resumed = MAPPOTrainer(inputs).train(
        output_dir=tmp_path / "resumed",
        resume_from=partial.checkpoint,
    )
    uninterrupted = MAPPOTrainer(inputs).train(output_dir=tmp_path / "uninterrupted")

    assert partial.progress.environment_steps == 8
    assert resumed.progress == uninterrupted.progress
    _assert_complete_checkpoints_equal(
        load_training_checkpoint(resumed.checkpoint),
        load_training_checkpoint(uninterrupted.checkpoint),
    )
    assert not list((tmp_path / "resumed").rglob("*.tmp"))


def test_saved_actor_loads_for_compatible_decentralized_evaluation(tmp_path: Path) -> None:
    inputs = _smoke_inputs()
    result = MAPPOTrainer(inputs).train(
        output_dir=tmp_path / "train",
        stop_after_environment_steps=8,
    )
    checkpoint = load_training_checkpoint(result.checkpoint)
    actor = actor_from_checkpoint(
        checkpoint,
        environment=inputs.environment,
        device=torch.device("cpu"),
    )

    assert actor.input_dim == checkpoint.metadata.signature.actor_input_dim
    assert actor.move_action_count == checkpoint.metadata.signature.move_action_count
    assert actor.message_action_count == checkpoint.metadata.signature.message_action_count
    assert not hasattr(actor, "critic")

    incompatible_environment = inputs.environment.model_copy(
        update={
            "communication": CommunicationConfig(
                enabled=False,
                vocabulary_size=inputs.environment.communication.vocabulary_size,
                budget_per_agent=0,
            )
        }
    )
    with pytest.raises(TrainingError, match="incompatible with the environment"):
        actor_from_checkpoint(
            checkpoint,
            environment=incompatible_environment,
            device=torch.device("cpu"),
        )

    changed_inputs = TrainingInputs(
        inputs.training.model_copy(update={"learning_rate": 0.0004}),
        inputs.environment,
        inputs.validation,
    )
    with pytest.raises(TrainingError, match="fingerprints do not match"):
        MAPPOTrainer(changed_inputs).train(
            output_dir=tmp_path / "invalid-resume",
            resume_from=result.checkpoint,
        )
    assert not (tmp_path / "invalid-resume").exists()


@pytest.mark.parametrize("target", [0, 3, 17])
def test_training_rejects_invalid_explicit_targets(target: int) -> None:
    trainer = MAPPOTrainer(_smoke_inputs())
    assert trainer.name == "shared-actor-centralized-critic-ppo"
    with pytest.raises(TrainingError, match="training target"):
        trainer._resolve_target(TrainingProgress(0, 0, 0), target)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("shape", "obstacle grid"),
        ("budgets", "budgets do not match"),
        ("messages", "messages do not match"),
        ("recovered", "not a subset"),
        ("bounds", "outside the world bounds"),
        ("obstacle", "on an obstacle"),
        ("negative_budget", "budgets cannot be negative"),
        ("negative_message", "messages cannot be negative"),
    ],
)
def test_world_checkpoint_rejects_invalid_simulator_state(case: str, message: str) -> None:
    raw: dict[str, Any] = {
        "schema_version": 1,
        "width": 2,
        "height": 2,
        "obstacles": [[False, False], [False, False]],
        "agent_positions": {"agent_0": {"row": 0, "col": 0}},
        "targets": [{"row": 1, "col": 1}],
        "recovered_targets": [],
        "communication_budgets": {"agent_0": 1},
        "latest_messages": {"agent_0": 0},
        "step_count": 0,
    }
    if case == "shape":
        raw["obstacles"] = [[False]]
    elif case == "budgets":
        raw["communication_budgets"] = {}
    elif case == "messages":
        raw["latest_messages"] = {}
    elif case == "recovered":
        raw["recovered_targets"] = [{"row": 0, "col": 1}]
    elif case == "bounds":
        raw["agent_positions"] = {"agent_0": {"row": 2, "col": 0}}
    elif case == "obstacle":
        raw["obstacles"] = [[True, False], [False, False]]
    elif case == "negative_budget":
        raw["communication_budgets"] = {"agent_0": -1}
    else:
        raw["latest_messages"] = {"agent_0": -1}
    with pytest.raises(ValidationError, match=message):
        WorldStateCheckpoint.model_validate(raw)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema", "invalid grid-rescue"),
        ("dimensions", "dimensions do not match"),
        ("world_agents", "world agents do not match"),
        ("live_agents", "unknown live agents"),
        ("step", "step count exceeds"),
        ("budget", "budget exceeds"),
        ("message", "message exceeds"),
    ],
)
def test_environment_restore_rejects_incompatible_checkpoint(
    case: str,
    message: str,
) -> None:
    inputs = _smoke_inputs()
    environment = GridRescueParallelEnv(inputs.environment)
    environment.reset(seed=7)
    raw = copy.deepcopy(environment.checkpoint_state())
    if case == "schema":
        raw = {"schema_version": 1}
    elif case == "dimensions":
        raw["world"]["width"] += 1
        raw["world"]["obstacles"] = [[*row, False] for row in raw["world"]["obstacles"]]
    elif case == "world_agents":
        removed = inputs.environment.num_agents - 1
        agent = f"agent_{removed}"
        raw["world"]["agent_positions"].pop(agent)
        raw["world"]["communication_budgets"].pop(agent)
        raw["world"]["latest_messages"].pop(agent)
    elif case == "live_agents":
        raw["agents"] = ["unknown"]
    elif case == "step":
        raw["world"]["step_count"] = inputs.environment.max_steps + 1
    elif case == "budget":
        raw["world"]["communication_budgets"]["agent_0"] = (
            inputs.environment.communication.budget_per_agent + 1
        )
    else:
        raw["world"]["latest_messages"]["agent_0"] = (
            inputs.environment.communication.vocabulary_size + 1
        )
    with pytest.raises(TrainingError, match=message):
        environment.restore_checkpoint_state(raw)
    environment.close()


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("actor_state", [], "actor_state"),
        ("optimizer_state", [], "optimizer state"),
        ("collector_state", [], "collector state"),
        ("training_records", {}, "training records"),
    ],
)
def test_checkpoint_loader_rejects_invalid_outer_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    inputs = _smoke_inputs()
    metadata = CheckpointMetadata(
        training_config=inputs.training,
        training_fingerprint=training_definition_fingerprint(inputs.training),
        environment_fingerprint=configuration_fingerprint(inputs.environment),
        validation_fingerprint=configuration_fingerprint(inputs.validation),
        signature=probe_learner_signature(inputs.environment),
        progress=CheckpointProgress.from_progress(TrainingProgress(0, 0, 0)),
        training_complete=False,
    )
    raw: dict[str, object] = {
        "metadata": metadata.model_dump(mode="json"),
        "actor_state": {},
        "critic_state": {},
        "optimizer_state": {},
        "collector_state": {},
        "training_records": [],
    }
    raw[field] = invalid
    path = tmp_path / f"invalid-{field}.pt"
    torch.save(raw, path)
    with pytest.raises(TrainingError, match=message):
        load_training_checkpoint(path)


def test_checkpoint_loader_and_hash_reject_missing_or_malformed_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(TrainingError, match="cannot load"):
        load_training_checkpoint(missing)
    with pytest.raises(TrainingError, match="cannot hash"):
        checkpoint_sha256(missing)

    not_mapping = tmp_path / "not-mapping.pt"
    torch.save([], not_mapping)
    with pytest.raises(TrainingError, match="must contain a mapping"):
        load_training_checkpoint(not_mapping)

    wrong_fields = tmp_path / "wrong-fields.pt"
    torch.save({"metadata": {}}, wrong_fields)
    with pytest.raises(TrainingError, match="fields do not match"):
        load_training_checkpoint(wrong_fields)


def test_initial_checkpoint_has_exact_declared_actor_identity(tmp_path: Path) -> None:
    inputs = _smoke_inputs()
    initial = MAPPOTrainer(inputs).initialize(output_dir=tmp_path / "initial")
    first = load_training_checkpoint(initial.checkpoint)
    second = load_training_checkpoint(
        MAPPOTrainer(inputs).initialize(output_dir=tmp_path / "second").checkpoint
    )

    assert initial.progress == TrainingProgress(0, 0, 0)
    assert model_state_sha256(first.actor_state) == model_state_sha256(second.actor_state)
    assert first.training_records == ()


def test_validation_selection_and_pathology_warnings_are_predeclared() -> None:
    base = {
        "success_rate": 0.0,
        "exploration_coverage": 0.4,
        "team_efficiency": 0.01,
        "duplicated_exploration": 0.5,
        "episode_length": 100.0,
    }
    improved = {**base, "exploration_coverage": 0.5}
    assert validation_selection_key(improved) > validation_selection_key(base)
    with pytest.raises(TrainingError, match="missing selection fields"):
        validation_selection_key({"success_rate": 1.0})

    diagnostics = PPOUpdateDiagnostics(
        total_loss=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        move_entropy=0.0,
        message_entropy=0.0,
        approximate_kl=0.3,
        clip_fraction=1.0,
        mean_probability_ratio=1.0,
        explained_variance=0.0,
        maximum_pre_clip_gradient_norm=0.0,
        maximum_post_clip_gradient_norm=0.0,
        valid_sample_count=1,
        minibatch_count=1,
        epoch_sample_orders=((0,),),
    )
    warnings = MAPPOTrainer(_smoke_inputs())._stability_warnings(
        diagnostics,
        move_frequencies=(1.0, 0.0, 0.0, 0.0, 0.0),
        communication_rate=0.0,
    )
    assert set(warnings) == {
        "near-zero-gradient",
        "excessive-approximate-kl",
        "all-samples-clipped",
        "movement-action-collapse",
        "communication-always-silent",
    }
