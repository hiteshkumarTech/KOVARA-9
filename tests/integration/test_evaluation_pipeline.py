import pytest

from kovara9.agents.random import RandomPolicy
from kovara9.config.models import (
    EnvConfig,
    EvaluationConfig,
    SeedPartitionsConfig,
    SeedRangeConfig,
)
from kovara9.evaluation.runner import evaluate_policy


@pytest.mark.integration
def test_evaluation_uses_every_explicit_seed(easy_config: EnvConfig) -> None:
    config = EvaluationConfig(
        name="integration",
        seeds=(20000, 20001),
        seed_partition="test",
        seed_partitions=SeedPartitionsConfig(
            train=SeedRangeConfig(start=0, count=10_000),
            validation=SeedRangeConfig(start=10_000, count=1_000),
            test=SeedRangeConfig(start=20_000, count=1_000),
        ),
        bootstrap_samples=10,
    )
    result = evaluate_policy(
        env_config=easy_config,
        evaluation_config=config,
        policy_factory=RandomPolicy,
    )
    assert [record.seed for record in result.records] == [20000, 20001]
    assert result.summary.episode_count == 2
    assert result.summary.policy == "random"
