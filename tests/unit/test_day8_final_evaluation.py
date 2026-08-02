from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import (
    configuration_fingerprint,
    load_evaluation_config,
    load_training_inputs,
)
from kovara9.config.models import (
    EnvConfig,
    EvaluationConfig,
    SeedPartitionsConfig,
    SeedRangeConfig,
)
from kovara9.core.errors import ArtifactError, ConfigurationError, TrainingError
from kovara9.core.types import Move, Position, WorldSnapshot
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.evaluation.runner import collision_blocked_agents, evaluate_policy
from kovara9.experiments import day8 as day8_module
from kovara9.experiments.day6 import (
    AlignedPolicyEvaluation,
    Day6SeedResult,
    aggregate_training_seed_metric,
    load_candidate_freeze,
    paired_metric_difference,
    validate_candidate_freeze,
)
from kovara9.experiments.day8 import (
    FINAL_POLICY_ORDER,
    checkpoint_integrity,
    claim_final_test,
    classify_final_performance,
    file_sha256,
    generalization_gap,
    optimizer_state_sha256,
    reject_consumed_test_partition,
    validate_policy_seed_alignment,
    verify_preregistration,
)
from kovara9.training.checkpoint import model_state_sha256, training_definition_fingerprint
from kovara9.training.config import NetworkConfig
from kovara9.training.encoding import ActorObservationEncoder
from kovara9.training.evaluation import actor_policy_factory
from kovara9.training.networks import CentralizedCritic, SharedActor

PARTITIONS = SeedPartitionsConfig(
    train=SeedRangeConfig(start=0, count=10),
    validation=SeedRangeConfig(start=10, count=10),
    test=SeedRangeConfig(start=20, count=10),
)
PREREGISTRATION_SHA256 = "cde300dcf8cc33f559bf5accaf99f541eafdfc57ed7559a421816a1fb3601b9a"


def _evaluation() -> EvaluationConfig:
    return EvaluationConfig(
        name="day8-integrity",
        seeds=(20,),
        seed_partition="test",
        seed_partitions=PARTITIONS,
        bootstrap_samples=0,
        bootstrap_confidence=0.95,
    )


def _metrics(
    *,
    success: float = 0.0,
    targets: float = 0.0,
    completion: float = 0.0,
    coverage: float = 0.0,
) -> dict[str, float]:
    return {
        "success_rate": success,
        "targets_recovered": targets,
        "completion_progress": completion,
        "exploration_coverage": coverage,
    }


def test_final_candidate_and_preregistration_are_frozen() -> None:
    candidate = Path("configs/training/mappo_final_candidate.yaml")
    freeze = load_candidate_freeze(Path("configs/training/mappo_final_candidate.freeze.json"))
    validate_candidate_freeze(candidate, freeze)
    preregistration = Path("docs/day8-final-evaluation-preregistration.json")
    assert verify_preregistration(preregistration, PREREGISTRATION_SHA256)["status"] == (
        "preregistered"
    )


def test_preregistration_mutation_is_rejected_after_claim(tmp_path: Path) -> None:
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text('{"status":"preregistered"}\n', encoding="utf-8")
    identity = file_sha256(preregistration)
    claim_final_test(tmp_path / "consumed.json", {"preregistration_sha256": identity})
    preregistration.write_text('{"status":"preregistered","amended":true}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fingerprint mismatch"):
        verify_preregistration(preregistration, identity)


def test_training_validation_and_final_partitions_do_not_overlap() -> None:
    evaluation = load_evaluation_config(Path("configs/evaluation/generalization.yaml"))
    training = {0, 1, 2}
    validation = set(range(10_000, 10_020))
    final = set(evaluation.resolved_seeds)
    assert not training.intersection(validation)
    assert not training.intersection(final)
    assert not validation.intersection(final)


def test_all_policies_receive_identical_aligned_test_seeds(easy_config: EnvConfig) -> None:
    evaluation = _evaluation()
    result = evaluate_policy(
        env_config=easy_config,
        evaluation_config=evaluation,
        policy_factory=RandomPolicy,
    )
    results = {policy: {"reference": result, "structural": result} for policy in FINAL_POLICY_ORDER}
    validate_policy_seed_alignment(results, evaluation.resolved_seeds)
    results["trained_seed_2"] = {"reference": result}
    with pytest.raises(TrainingError, match="environments are incomplete"):
        validate_policy_seed_alignment(results, evaluation.resolved_seeds)


def test_paired_results_reject_test_seed_reordering() -> None:
    left = AlignedPolicyEvaluation(
        root_seed=0,
        policy="trained",
        seed_partition="test",
        evaluation_seeds=(20_000, 20_001),
        metrics={"success_rate": (0.0, 1.0)},
    )
    right = AlignedPolicyEvaluation(
        root_seed=0,
        policy="untrained",
        seed_partition="test",
        evaluation_seeds=(20_001, 20_000),
        metrics={"success_rate": (1.0, 0.0)},
    )
    with pytest.raises(TrainingError, match="different evaluation seeds"):
        paired_metric_difference(left, right, "success_rate")


def test_neural_evaluation_does_not_change_actor_critic_or_optimizer(
    easy_config: EnvConfig,
) -> None:
    environment = GridRescueParallelEnv(easy_config)
    agent = environment.possible_agents[0]
    encoder = ActorObservationEncoder(environment.observation_space(agent))
    environment.close()
    network = NetworkConfig(actor_hidden_sizes=(8,), critic_hidden_sizes=(8,), activation="tanh")
    actor = SharedActor(
        input_dim=encoder.input_dim,
        move_action_count=encoder.move_action_count,
        message_action_count=encoder.message_action_count,
        config=network,
        seed=4,
    )
    critic = CentralizedCritic(input_dim=1, config=network, seed=5)
    optimizer = {"state": {0: {"step": torch.tensor(1), "exp_avg": torch.ones(2)}}, "groups": []}
    before = (
        model_state_sha256(actor.state_dict()),
        model_state_sha256(critic.state_dict()),
        optimizer_state_sha256(optimizer),
    )
    factory = actor_policy_factory(
        actor=actor,
        device=torch.device("cpu"),
        policy_name="integrity-actor",
        parameters={"deterministic": True},
    )
    evaluate_policy(env_config=easy_config, evaluation_config=_evaluation(), policy_factory=factory)
    after = (
        model_state_sha256(actor.state_dict()),
        model_state_sha256(critic.state_dict()),
        optimizer_state_sha256(optimizer),
    )
    assert after == before


def test_checkpoint_file_checksum_is_unchanged_by_integrity_read(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable-checkpoint")
    before = file_sha256(checkpoint)
    assert checkpoint.read_bytes() == b"immutable-checkpoint"
    assert file_sha256(checkpoint) == before


def test_final_test_claim_is_exclusive(tmp_path: Path) -> None:
    consumed = tmp_path / "final_test_consumed.json"
    claim_final_test(consumed, {"status": "in_progress"})
    with pytest.raises(ArtifactError, match="already claimed"):
        claim_final_test(consumed, {"status": "in_progress"})


def test_tuning_workflow_rejects_consumed_test_partition(tmp_path: Path) -> None:
    consumed = tmp_path / "final_test_consumed.json"
    consumed.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="partition is consumed"):
        reject_consumed_test_partition(_evaluation(), consumed)


def test_generalization_gap_uses_validation_minus_held_out() -> None:
    assert generalization_gap(0.75, 0.50) == pytest.approx(0.25)
    assert generalization_gap(0.25, 0.50) == pytest.approx(-0.25)
    with pytest.raises(ValueError, match="finite"):
        generalization_gap(float("nan"), 0.0)


def test_aggregate_statistics_require_all_three_training_seeds() -> None:
    complete = [
        Day6SeedResult(root_seed=seed, status="complete", metrics={"success_rate": 0.0})
        for seed in (0, 1, 2)
    ]
    assert aggregate_training_seed_metric(complete, "success_rate").count == 3
    with pytest.raises(TrainingError, match="identities do not match"):
        aggregate_training_seed_metric(complete[:2], "success_rate")


@pytest.mark.parametrize(
    ("trained", "untrained", "random", "valid", "expected"),
    [
        (
            {seed: _metrics(success=0.6, targets=3.0, completion=0.8) for seed in (0, 1, 2)},
            {seed: _metrics(success=0.1, targets=1.0, completion=0.2) for seed in (0, 1, 2)},
            _metrics(success=0.4, targets=2.0, completion=0.5),
            True,
            "reproducible_task_learning_and_transfer",
        ),
        (
            {seed: _metrics(completion=0.1) for seed in (0, 1, 2)},
            {seed: _metrics(completion=0.2) for seed in (0, 1, 2)},
            _metrics(),
            True,
            "negative_transfer",
        ),
        (
            {seed: _metrics(coverage=0.5) for seed in (0, 1, 2)},
            {seed: _metrics(coverage=0.2) for seed in (0, 1, 2)},
            _metrics(),
            True,
            "exploration_transfer_without_task_completion",
        ),
        (
            {0: _metrics(targets=1.0), 1: _metrics(), 2: _metrics()},
            {seed: _metrics() for seed in (0, 1, 2)},
            _metrics(),
            True,
            "weak_or_inconsistent_transfer",
        ),
        (
            {seed: {**_metrics(), "shared_return": float("nan")} for seed in (0, 1, 2)},
            {seed: _metrics() for seed in (0, 1, 2)},
            _metrics(),
            True,
            "no_meaningful_transfer",
        ),
        (
            {seed: _metrics() for seed in (0, 1)},
            {seed: _metrics() for seed in (0, 1, 2)},
            _metrics(),
            False,
            "evaluation_failure",
        ),
    ],
)
def test_final_classification_follows_preregistered_metrics(
    trained: dict[int, dict[str, float]],
    untrained: dict[int, dict[str, float]],
    random: dict[str, float],
    valid: bool,
    expected: str,
) -> None:
    assert classify_final_performance(trained, untrained, random, valid=valid) == expected


def test_collision_diagnostic_separates_agent_contention_from_wall_blocks() -> None:
    snapshot = WorldSnapshot(
        width=5,
        height=5,
        obstacles=np.zeros((5, 5), dtype=np.bool_),
        agent_positions={"agent_0": Position(1, 1), "agent_1": Position(1, 2)},
        targets=frozenset(),
        recovered_targets=frozenset(),
        communication_budgets={"agent_0": 0, "agent_1": 0},
        latest_messages={"agent_0": 0, "agent_1": 0},
        step_count=0,
    )
    swap = {
        "agent_0": {"move": int(Move.EAST), "message": 0},
        "agent_1": {"move": int(Move.WEST), "message": 0},
    }
    assert collision_blocked_agents(snapshot, swap, {"agent_0", "agent_1"}) == {
        "agent_0",
        "agent_1",
    }
    wall = {"agent_0": {"move": int(Move.NORTH), "message": 0}}
    assert collision_blocked_agents(snapshot, wall, {"agent_0"}) == set()


def test_checkpoint_integrity_requires_a_valid_checkpoint(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a checkpoint")
    with pytest.raises(TrainingError, match="cannot load checkpoint"):
        checkpoint_integrity(invalid)


def test_final_integrity_helpers_reject_malformed_inputs(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="cannot hash"):
        file_sha256(tmp_path / "missing.json")
    with pytest.raises(TrainingError, match="cannot fingerprint optimizer value"):
        optimizer_state_sha256({"invalid": object()})
    draft = tmp_path / "draft.json"
    draft.write_text('{"status":"draft"}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not preregistered"):
        verify_preregistration(draft, file_sha256(draft))
    trained = {seed: _metrics() for seed in (0, 1, 2)}
    trained[0]["success_rate"] = float("nan")
    assert (
        classify_final_performance(
            trained,
            {seed: _metrics() for seed in (0, 1, 2)},
            _metrics(),
        )
        == "evaluation_failure"
    )


def test_final_orchestrator_completes_all_policies_and_locks_once(
    tmp_path: Path,
    easy_config: EnvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = load_training_inputs(Path("configs/training/mappo_final_candidate.yaml"))
    evaluation = _evaluation()
    reference = easy_config.model_copy(update={"max_steps": 1})
    structural = easy_config.model_copy(update={"max_steps": 2})
    candidate_environment_identity = "a" * 64
    validation_identity = "b" * 64
    identity = {
        "checkpoint_sha256": "c" * 64,
        "actor_state_sha256": "d" * 64,
        "critic_state_sha256": "e" * 64,
        "optimizer_state_sha256": "f" * 64,
    }
    policies: list[dict[str, Any]] = [
        {"id": "random", "checkpoint": None},
        {"id": "frontier", "checkpoint": None},
    ]
    steps = {0: 11, 1: 12, 2: 13}
    for kind in ("untrained", "trained"):
        for seed in (0, 1, 2):
            policy = f"{kind}_seed_{seed}"
            spec: dict[str, Any] = {
                "id": policy,
                "training_seed": seed,
                "checkpoint": f"checkpoints/{policy}.pt",
                **{name: identity[name] for name in identity if name != "optimizer_state_sha256"},
            }
            if kind == "trained":
                spec["selected_environment_steps"] = steps[seed]
            policies.append(spec)
    preregistration = {
        "status": "preregistered",
        "seed_partitions": {"final_test_seeds": [20]},
        "candidate": {
            "configuration_fingerprint": "1" * 64,
            "environment_fingerprint": candidate_environment_identity,
            "validation_fingerprint": validation_identity,
        },
        "evaluation": {
            "configuration_fingerprint": configuration_fingerprint(evaluation),
            "reference_environment": {"fingerprint": configuration_fingerprint(reference)},
            "structural_environment": {"fingerprint": configuration_fingerprint(structural)},
        },
        "policies": policies,
    }

    def fake_checkpoint(path: Path) -> SimpleNamespace:
        policy = path.stem
        seed = int(policy.rsplit("_", maxsplit=1)[1])
        expected_training = inputs.training.model_copy(update={"seed": seed})
        environment_steps = 0 if policy.startswith("untrained") else steps[seed]
        return SimpleNamespace(
            metadata=SimpleNamespace(
                training_config=expected_training,
                training_fingerprint=training_definition_fingerprint(expected_training),
                environment_fingerprint=candidate_environment_identity,
                validation_fingerprint=validation_identity,
                progress=SimpleNamespace(environment_steps=environment_steps),
            ),
            actor_state={},
        )

    monkeypatch.setattr(day8_module, "verify_preregistration", lambda *_args: preregistration)
    monkeypatch.setattr(day8_module, "load_training_inputs", lambda _path: inputs)
    monkeypatch.setattr(day8_module, "load_candidate_freeze", lambda _path: object())
    monkeypatch.setattr(day8_module, "validate_candidate_freeze", lambda *_args: None)
    monkeypatch.setattr(day8_module, "load_evaluation_config", lambda _path: evaluation)
    monkeypatch.setattr(day8_module, "reject_consumed_test_partition", lambda *_args: None)
    monkeypatch.setattr(
        day8_module,
        "load_comparison_environment_configs",
        lambda _evaluation: (reference, structural),
    )
    monkeypatch.setattr(day8_module, "load_training_checkpoint", fake_checkpoint)
    monkeypatch.setattr(day8_module, "checkpoint_integrity", lambda *_args: identity)
    monkeypatch.setattr(day8_module, "model_state_sha256", lambda _state: "initial")
    monkeypatch.setattr(
        day8_module,
        "untrained_actor_from_checkpoint_definition",
        lambda *_args, **_kwargs: SimpleNamespace(state_dict=dict),
    )
    monkeypatch.setattr(day8_module, "_git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(
        day8_module,
        "_policy_factory_for_checkpoint",
        lambda *_args, **_kwargs: RandomPolicy,
    )
    monkeypatch.setattr(
        day8_module,
        "_evaluate_pair",
        lambda **_kwargs: {"reference": object(), "structural": object()},
    )
    monkeypatch.setattr(day8_module, "validate_policy_seed_alignment", lambda *_args: None)

    consumed = tmp_path / "final_test_consumed.json"
    output = tmp_path / "output"
    record = day8_module.run_final_evaluation(
        candidate_path=tmp_path / "candidate.yaml",
        freeze_path=tmp_path / "freeze.json",
        evaluation_path=tmp_path / "evaluation.yaml",
        preregistration_path=tmp_path / "preregistration.json",
        preregistration_sha256="0" * 64,
        artifact_root=tmp_path,
        output=output,
        consumed_record=consumed,
        device_name="cpu",
    )
    assert record["status"] == "complete"
    assert record["completed_policies"] == list(FINAL_POLICY_ORDER)
    assert record["checkpoint_integrity_before"] == record["checkpoint_integrity_after"]
    with pytest.raises(ArtifactError, match="output already exists"):
        day8_module.run_final_evaluation(
            candidate_path=tmp_path / "candidate.yaml",
            freeze_path=tmp_path / "freeze.json",
            evaluation_path=tmp_path / "evaluation.yaml",
            preregistration_path=tmp_path / "preregistration.json",
            preregistration_sha256="0" * 64,
            artifact_root=tmp_path,
            output=output,
            consumed_record=consumed,
            device_name="cpu",
        )


def test_evaluate_pair_writes_both_environment_suites(
    tmp_path: Path, easy_config: EnvConfig
) -> None:
    reference = easy_config.model_copy(update={"max_steps": 1})
    structural = easy_config.model_copy(update={"max_steps": 2})
    results = day8_module._evaluate_pair(
        policy_factory=RandomPolicy,
        evaluation=_evaluation(),
        reference=reference,
        structural=structural,
        output=tmp_path / "pair",
    )
    assert tuple(results) == ("reference", "structural")
    assert (tmp_path / "pair/episodes.jsonl").is_file()
    assert (tmp_path / "pair/held_out_episodes.jsonl").is_file()


def test_checkpoint_policy_factory_records_frozen_identity(
    easy_config: EnvConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = GridRescueParallelEnv(easy_config)
    agent = environment.possible_agents[0]
    encoder = ActorObservationEncoder(environment.observation_space(agent))
    environment.close()
    actor = SharedActor(
        input_dim=encoder.input_dim,
        move_action_count=encoder.move_action_count,
        message_action_count=encoder.message_action_count,
        config=NetworkConfig(actor_hidden_sizes=(8,), critic_hidden_sizes=(8,), activation="tanh"),
        seed=1,
    )
    loaded = SimpleNamespace(
        metadata=SimpleNamespace(
            training_config=SimpleNamespace(seed=2),
            progress=SimpleNamespace(environment_steps=3),
        )
    )
    monkeypatch.setattr(day8_module, "actor_from_checkpoint", lambda *_args, **_kwargs: actor)
    factory = day8_module._policy_factory_for_checkpoint(
        "trained_seed_2",
        loaded,  # type: ignore[arg-type]
        easy_config,
        torch.device("cpu"),
        {"checkpoint_sha256": "a" * 64},
    )
    policy = factory()
    assert policy.name == "trained_seed_2"
    assert policy.parameters["environment_steps"] == 3
