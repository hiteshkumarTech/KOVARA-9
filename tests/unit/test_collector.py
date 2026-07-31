from collections.abc import Callable

import torch

from kovara9.config.models import CommunicationConfig, EnvConfig
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.training.collector import RolloutCollection, SynchronousRolloutCollector
from kovara9.training.config import NetworkConfig
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.seeds import ExperimentSeedStreams


def _network_config() -> NetworkConfig:
    return NetworkConfig(
        actor_hidden_sizes=(16,),
        critic_hidden_sizes=(16,),
        activation="tanh",
    )


def _collect(  # noqa: PLR0913
    config: EnvConfig,
    *,
    root_seed: int,
    num_environments: int = 1,
    rollout_length: int = 4,
    deterministic: bool = False,
    prepare_actor: Callable[[SharedActor], None] | None = None,
) -> RolloutCollection:
    probe = GridRescueParallelEnv(config)
    actor_encoder = ActorObservationEncoder(probe.observation_space(probe.possible_agents[0]))
    critic_encoder = CentralStateEncoder(probe.state_space)
    streams = ExperimentSeedStreams(root_seed)
    actor = SharedActor(
        input_dim=actor_encoder.input_dim,
        move_action_count=actor_encoder.move_action_count,
        message_action_count=actor_encoder.message_action_count,
        config=_network_config(),
        seed=streams.actor_initialization,
    )
    critic = CentralizedCritic(
        input_dim=critic_encoder.input_dim,
        config=_network_config(),
        seed=streams.critic_initialization,
    )
    probe.close()
    if prepare_actor is not None:
        prepare_actor(actor)
    collector = SynchronousRolloutCollector(
        environment_factory=lambda: GridRescueParallelEnv(config),
        num_environments=num_environments,
        rollout_length=rollout_length,
        actor=actor,
        critic=critic,
        root_seed=root_seed,
        device=torch.device("cpu"),
    )
    try:
        return collector.collect(deterministic=deterministic)
    finally:
        collector.close()


def _tensor_fields(collection: RolloutCollection) -> tuple[torch.Tensor, ...]:
    batch = collection.batch
    return (
        batch.actor_features,
        batch.critic_features,
        batch.move_action_masks,
        batch.message_action_masks,
        batch.move_actions,
        batch.message_actions,
        batch.joint_log_probabilities,
        batch.rewards,
        batch.values,
        batch.next_values,
        batch.terminated,
        batch.truncated,
        batch.episode_starts,
        batch.active_agents,
        batch.communication_rejections,
        batch.environment_ids,
        batch.transition_ids,
    )


def test_single_and_multiple_environment_rollout_shapes(easy_config: EnvConfig) -> None:
    single = _collect(easy_config, root_seed=4, rollout_length=3).batch
    assert single.actor_features.shape[:3] == (3, 1, easy_config.num_agents)
    assert single.critic_features.shape[:2] == (3, 1)
    assert single.move_actions.shape == (3, 1, easy_config.num_agents)
    assert single.environment_ids.tolist() == [[0], [0], [0]]

    multiple = _collect(
        easy_config,
        root_seed=4,
        num_environments=2,
        rollout_length=3,
    ).batch
    assert multiple.actor_features.shape[:3] == (3, 2, easy_config.num_agents)
    assert multiple.critic_features.shape[:2] == (3, 2)
    assert multiple.environment_ids.tolist() == [[0, 1], [0, 1], [0, 1]]
    assert multiple.agent_order == ("agent_0", "agent_1")


def test_equal_root_seeds_reproduce_complete_rollouts(easy_config: EnvConfig) -> None:
    first = _collect(easy_config, root_seed=17, num_environments=2)
    second = _collect(easy_config, root_seed=17, num_environments=2)
    assert first.reset_seeds == second.reset_seeds
    assert first.completed_episodes == second.completed_episodes
    assert all(
        torch.equal(first_tensor, second_tensor)
        for first_tensor, second_tensor in zip(
            _tensor_fields(first), _tensor_fields(second), strict=True
        )
    )


def test_different_root_seeds_change_collected_trajectory(easy_config: EnvConfig) -> None:
    first = _collect(easy_config, root_seed=17)
    second = _collect(easy_config, root_seed=18)
    assert first.reset_seeds != second.reset_seeds
    assert not torch.equal(first.batch.actor_features, second.batch.actor_features)


def test_collector_respects_every_environment_action_mask(easy_config: EnvConfig) -> None:
    batch = _collect(easy_config, root_seed=8, rollout_length=12).batch
    selected_moves = torch.gather(
        batch.move_action_masks,
        -1,
        batch.move_actions.unsqueeze(-1),
    )
    selected_messages = torch.gather(
        batch.message_action_masks,
        -1,
        batch.message_actions.unsqueeze(-1),
    )
    assert bool(selected_moves.all())
    assert bool(selected_messages.all())
    assert not bool(batch.communication_rejections.any())


def test_communication_disabled_never_samples_a_message(easy_config: EnvConfig) -> None:
    disabled = easy_config.model_copy(
        update={
            "communication": CommunicationConfig(
                enabled=False,
                vocabulary_size=easy_config.communication.vocabulary_size,
                budget_per_agent=0,
            )
        }
    )
    batch = _collect(disabled, root_seed=2, rollout_length=8).batch
    assert batch.message_action_masks.shape[-1] == 1
    assert not bool(batch.message_actions.any())
    assert not bool(batch.communication_rejections.any())


def test_budget_exhaustion_forces_silent_valid_actions(easy_config: EnvConfig) -> None:
    limited = easy_config.model_copy(
        update={
            "communication": easy_config.communication.model_copy(update={"budget_per_agent": 1})
        }
    )

    def prefer_message(actor: SharedActor) -> None:
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor.message_head.bias[1] = 10.0

    batch = _collect(
        limited,
        root_seed=12,
        rollout_length=6,
        deterministic=True,
        prepare_actor=prefer_message,
    ).batch
    assert batch.message_actions[0].eq(1).all()
    exhausted_rows = ~batch.message_action_masks[..., 1:].any(dim=-1)
    assert bool(batch.message_actions[exhausted_rows].eq(0).all())
    assert not bool(batch.communication_rejections.any())


def test_terminal_boundaries_reset_with_next_per_environment_seed(
    easy_config: EnvConfig,
) -> None:
    one_step = easy_config.model_copy(update={"max_steps": 1})
    collection = _collect(
        one_step,
        root_seed=21,
        num_environments=2,
        rollout_length=3,
        deterministic=True,
    )
    batch = collection.batch
    assert bool(torch.logical_xor(batch.terminated, batch.truncated).all())
    assert bool(batch.episode_starts.all())
    assert len(collection.completed_episodes) == 6
    streams = ExperimentSeedStreams(21)
    assert collection.reset_seeds == tuple(
        tuple(streams.environment_reset(environment_id, episode) for episode in range(4))
        for environment_id in range(2)
    )
    assert batch.transition_ids.tolist() == [[0, 0], [1, 1], [2, 2]]


def test_rollout_is_finite_and_keeps_actor_and_critic_features_separate(
    easy_config: EnvConfig,
) -> None:
    batch = _collect(easy_config, root_seed=25, rollout_length=3).batch
    floating = [
        tensor
        for tensor in _tensor_fields(_collect(easy_config, root_seed=25))
        if tensor.is_floating_point()
    ]
    assert all(bool(torch.isfinite(tensor).all()) for tensor in floating)
    assert batch.actor_features.ndim == 4
    assert batch.critic_features.ndim == 3
    assert batch.actor_features.shape[-1] != batch.critic_features.shape[-1]
