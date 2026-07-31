import pytest

from kovara9.training.seeds import ExperimentSeedStreams


def test_semantic_seed_streams_are_repeatable_and_distinct() -> None:
    first = ExperimentSeedStreams(31)
    second = ExperimentSeedStreams(31)
    assert first.actor_initialization == second.actor_initialization
    assert first.critic_initialization == second.critic_initialization
    assert first.policy_sampling == second.policy_sampling
    assert first.environment_reset(0, 0) == second.environment_reset(0, 0)
    assert (
        len(
            {
                first.actor_initialization,
                first.critic_initialization,
                first.policy_sampling,
                first.environment_reset(0, 0),
                first.environment_reset(0, 1),
                first.environment_reset(1, 0),
                first.evaluation(0),
            }
        )
        == 7
    )


def test_environment_reset_seed_depends_on_explicit_instance_and_episode() -> None:
    streams = ExperimentSeedStreams(7)
    in_forward_order = [
        streams.environment_reset(environment_id, episode_index)
        for environment_id in range(2)
        for episode_index in range(3)
    ]
    in_reverse_order = {
        (environment_id, episode_index): streams.environment_reset(environment_id, episode_index)
        for episode_index in reversed(range(3))
        for environment_id in reversed(range(2))
    }
    assert in_forward_order == [
        in_reverse_order[(environment_id, episode_index)]
        for environment_id in range(2)
        for episode_index in range(3)
    ]


def test_seed_streams_reject_negative_identifiers() -> None:
    with pytest.raises(ValueError, match="root"):
        ExperimentSeedStreams(-1)
    streams = ExperimentSeedStreams(0)
    with pytest.raises(ValueError, match="environment_id"):
        streams.environment_reset(-1, 0)
    with pytest.raises(ValueError, match="episode_index"):
        streams.environment_reset(0, -1)
