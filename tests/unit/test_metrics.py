import pytest

from kovara9.config.models import EvaluationConfig, SeedPartitionsConfig, SeedRangeConfig
from kovara9.evaluation.metrics import (
    aggregate_records,
    duplicated_exploration,
    exploration_coverage,
    generalization_gap,
    team_efficiency,
)
from kovara9.evaluation.records import EpisodeRecord

PARTITIONS = SeedPartitionsConfig(
    train=SeedRangeConfig(start=0, count=10),
    validation=SeedRangeConfig(start=10, count=10),
    test=SeedRangeConfig(start=20, count=10),
)


def _record(seed: int, success: bool) -> EpisodeRecord:
    return EpisodeRecord(
        seed=seed,
        success=success,
        episode_length=10,
        targets_recovered=int(success),
        total_targets=1,
        exploration_coverage=0.5,
        duplicated_exploration=0.25,
        communication_messages=2,
        messages_per_agent_step=0.1,
        team_efficiency=0.05,
        shared_return=1.0,
        termination_reason="success" if success else "time_limit",
    )


def test_hand_computed_spatial_metrics() -> None:
    observed = [{1, 2, 3}, {3, 4}]
    assert exploration_coverage(observed, 8) == 0.5
    assert duplicated_exploration(observed) == 0.2
    assert duplicated_exploration([set(), set()]) == 0
    assert team_efficiency(3, 12) == 0.25
    assert team_efficiency(0, 0) == 0


def test_aggregation_and_generalization_are_deterministic() -> None:
    records = [_record(20, True), _record(21, False)]
    config = EvaluationConfig(
        name="test",
        seeds=(20, 21),
        seed_partition="test",
        seed_partitions=PARTITIONS,
        bootstrap_samples=100,
        bootstrap_confidence=0.95,
    )
    first = aggregate_records(records, config, "policy")
    second = aggregate_records(records, config, "policy")
    assert first == second
    assert first.metrics["success_rate"].mean == 0.5
    assert first.metrics["episode_length"].standard_deviation == 0
    assert generalization_gap(records, [_record(3, False)]) == 0.5


def test_aggregation_without_bootstrap_has_no_interval() -> None:
    config = EvaluationConfig(
        name="one",
        seeds=(20,),
        seed_partition="test",
        seed_partitions=PARTITIONS,
        bootstrap_samples=0,
    )
    summary = aggregate_records([_record(20, True)], config, "policy")
    metric = summary.metrics["success_rate"]
    assert metric.confidence_low is None
    assert metric.confidence_high is None


def test_aggregation_rejects_result_config_seed_mismatch() -> None:
    config = EvaluationConfig(
        name="mismatch",
        seeds=(20,),
        seed_partition="test",
        seed_partitions=PARTITIONS,
        bootstrap_samples=0,
    )
    with pytest.raises(ValueError, match="do not match configured seeds"):
        aggregate_records([_record(21, True)], config, "policy")
