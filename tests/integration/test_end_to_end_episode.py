import pytest

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.random import RandomPolicy
from kovara9.config.models import EnvConfig
from kovara9.evaluation.runner import run_episode


@pytest.mark.integration
@pytest.mark.parametrize("policy", [RandomPolicy, FrontierPolicy])
def test_baselines_complete_bounded_episodes(
    easy_config: EnvConfig,
    policy: type[RandomPolicy] | type[FrontierPolicy],
) -> None:
    record = run_episode(
        env_config=easy_config,
        seed=20000,
        policy_factory=policy,
    )
    assert 1 <= record.episode_length <= easy_config.max_steps
    assert 0 <= record.exploration_coverage <= 1
    assert 0 <= record.duplicated_exploration <= 1
    assert record.termination_reason in {"success", "time_limit"}
    assert record.targets_recovered <= record.total_targets
