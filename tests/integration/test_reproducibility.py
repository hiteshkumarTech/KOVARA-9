import pytest

from kovara9.agents.random import RandomPolicy
from kovara9.config.models import EnvConfig
from kovara9.evaluation.runner import run_episode


@pytest.mark.integration
def test_full_seeded_episode_record_repeats(easy_config: EnvConfig) -> None:
    first = run_episode(
        env_config=easy_config,
        seed=20007,
        policy_factory=RandomPolicy,
    )
    second = run_episode(
        env_config=easy_config,
        seed=20007,
        policy_factory=RandomPolicy,
    )
    assert first == second
