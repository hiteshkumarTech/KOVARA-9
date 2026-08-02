import json
from pathlib import Path

import pytest
import yaml

from kovara9.config.loader import configuration_fingerprint, load_training_inputs
from kovara9.core.errors import ConfigurationError, TrainingError
from kovara9.experiments.day6 import (
    DAY6_ROOT_SEEDS,
    AlignedPolicyEvaluation,
    CandidateEvidence,
    CandidateFreezeRecord,
    Day6SeedResult,
    aggregate_training_seed_metric,
    candidate_configuration_fingerprint,
    load_candidate_freeze,
    load_day6_training_inputs,
    paired_metric_difference,
    reward_fingerprint,
    select_validation_candidate,
    training_configuration_differences,
    validate_candidate_freeze,
)
from kovara9.training.checkpoint import training_definition_fingerprint


def _seed_result(root_seed: int, value: float) -> Day6SeedResult:
    return Day6SeedResult(
        root_seed=root_seed,
        status="complete",
        metrics={"success_rate": value},
    )


def _evaluation(
    root_seed: int,
    policy: str,
    values: tuple[float, ...],
) -> AlignedPolicyEvaluation:
    return AlignedPolicyEvaluation(
        root_seed=root_seed,
        policy=policy,
        seed_partition="validation",
        evaluation_seeds=(10_000, 10_001, 10_002),
        metrics={"success_rate": values},
    )


def _candidate(name: str, success: float, *, partition: str = "validation") -> CandidateEvidence:
    return CandidateEvidence.model_validate(
        {
            "name": name,
            "source_configuration": f"configs/training/{name}.yaml",
            "seed_partition": partition,
            "training_seeds": DAY6_ROOT_SEEDS,
            "metrics": {
                "success_rate": success,
                "targets_recovered": 1.0,
                "team_efficiency": 0.1,
                "exploration_coverage": 0.5,
                "duplicated_exploration": 0.2,
                "episode_length": 100.0,
            },
        }
    )


def test_multiseed_records_preserve_identity_and_aggregate_every_seed() -> None:
    records = [_seed_result(2, 0.3), _seed_result(0, 0.1), _seed_result(1, 0.2)]
    summary = aggregate_training_seed_metric(records, "success_rate")
    assert {record.root_seed for record in records} == set(DAY6_ROOT_SEEDS)
    assert summary.count == 3
    assert summary.mean == pytest.approx(0.2)
    assert summary.standard_deviation == pytest.approx(0.1)
    assert summary.minimum == 0.1
    assert summary.maximum == 0.3


def test_aggregate_rejects_missing_duplicate_and_failed_seeds() -> None:
    complete = [_seed_result(seed, 0.0) for seed in DAY6_ROOT_SEEDS]
    with pytest.raises(TrainingError, match="identities do not match"):
        aggregate_training_seed_metric(complete[:-1], "success_rate")
    with pytest.raises(TrainingError, match="duplicate"):
        aggregate_training_seed_metric([*complete[:-1], complete[0]], "success_rate")
    failed = Day6SeedResult(root_seed=2, status="failed", failure="numerical failure")
    with pytest.raises(TrainingError, match="failed Day 6 seeds"):
        aggregate_training_seed_metric([*complete[:2], failed], "success_rate")


def test_paired_differences_preserve_root_and_evaluation_seed_alignment() -> None:
    trained = _evaluation(1, "trained", (1.0, 0.0, 1.0))
    untrained = _evaluation(1, "untrained", (0.0, 0.0, 1.0))
    difference = paired_metric_difference(trained, untrained, "success_rate")
    assert difference.root_seed == 1
    assert difference.evaluation_seeds == (10_000, 10_001, 10_002)
    assert difference.values == (1.0, 0.0, 0.0)
    assert difference.summary.mean == pytest.approx(1 / 3)

    misaligned = untrained.model_copy(update={"evaluation_seeds": (10_000, 10_002, 10_001)})
    with pytest.raises(TrainingError, match="different evaluation seeds"):
        paired_metric_difference(trained, misaligned, "success_rate")
    wrong_root = untrained.model_copy(update={"root_seed": 2})
    with pytest.raises(TrainingError, match="different root seeds"):
        paired_metric_difference(trained, wrong_root, "success_rate")


def test_candidate_selection_uses_validation_metrics_only() -> None:
    longer = _candidate("mappo_day6_longer", 0.1)
    entropy = _candidate("mappo_day6_entropy", 0.2)
    assert select_validation_candidate([longer, entropy]) == entropy
    test_evidence = _candidate("forbidden", 1.0, partition="test")
    with pytest.raises(TrainingError, match="validation metrics only"):
        select_validation_candidate([longer, test_evidence])


def test_day6_config_changes_only_exposure_and_preserves_reward() -> None:
    day5 = load_training_inputs(Path("configs/training/mappo_day5_short.yaml"))
    day6 = load_training_inputs(Path("configs/training/mappo_day6_longer.yaml"))
    assert training_configuration_differences(day5.training, day6.training) == (
        "total_environment_steps",
    )
    assert reward_fingerprint(day5.environment) == reward_fingerprint(day6.environment)
    assert configuration_fingerprint(day5.environment) == configuration_fingerprint(
        day6.environment
    )


def test_day6_loader_rejects_test_partition(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/training/mappo_day6_longer.yaml").read_text(encoding="utf-8")
    )
    raw["environment_config"] = str(Path("configs/environments/grid_rescue_medium.yaml").resolve())
    raw["validation_config"] = str(Path("configs/evaluation/benchmark.yaml").resolve())
    path = tmp_path / "forbidden-test-training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="validation seed partition"):
        load_day6_training_inputs(path, root_seed=0)


def test_candidate_fingerprint_is_stable_and_mutation_is_detectable(tmp_path: Path) -> None:
    candidate = Path("configs/training/mappo_day6_longer.yaml")
    inputs = load_training_inputs(candidate)
    first = candidate_configuration_fingerprint(candidate)
    second = candidate_configuration_fingerprint(candidate)
    assert first == second

    raw = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    raw["environment_config"] = str(inputs.training.environment_config)
    raw["validation_config"] = str(inputs.training.validation_config)
    raw["learning_rate"] = 0.0004
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert candidate_configuration_fingerprint(mutated) != first

    freeze = CandidateFreezeRecord(
        source_configuration="configs/training/mappo_day6_longer.yaml",
        selection_metric="aggregate_validation_success_rate",
        selection_reason="controlled validation evidence",
        alternatives_evaluated=("mappo_day6_longer.yaml",),
        configuration_fingerprint=first,
        reward_fingerprint=reward_fingerprint(inputs.environment),
        environment_fingerprint=configuration_fingerprint(inputs.environment),
        training_seeds=DAY6_ROOT_SEEDS,
        validation_seeds=inputs.validation.resolved_seeds,
        git_commit="b7d8758",
    )
    validate_candidate_freeze(candidate, freeze)
    with pytest.raises(ConfigurationError, match="identity does not match"):
        validate_candidate_freeze(mutated, freeze)

    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze.model_dump(mode="json")), encoding="utf-8")
    assert load_candidate_freeze(freeze_path) == freeze
    assert training_definition_fingerprint(inputs.training) == first
