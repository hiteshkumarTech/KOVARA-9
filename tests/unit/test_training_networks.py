from typing import cast

import pytest
import torch

from kovara9.config.models import EnvConfig
from kovara9.core.errors import ConfigurationError, NumericalError, TrainingError
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.training.config import NetworkConfig
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.networks import (
    ActorInput,
    CentralizedCritic,
    CriticInput,
    SharedActor,
)
from kovara9.training.runtime import (
    configure_deterministic_algorithms,
    make_torch_generator,
    resolve_device,
)


def _network_config() -> NetworkConfig:
    return NetworkConfig(
        actor_hidden_sizes=(32, 16),
        critic_hidden_sizes=(32, 16),
        activation="tanh",
    )


def test_shared_actor_and_centralized_critic_shapes(easy_config: EnvConfig) -> None:
    env = GridRescueParallelEnv(easy_config)
    observations, _infos = env.reset(seed=3)
    device = torch.device("cpu")
    actor_encoder = ActorObservationEncoder(env.observation_space("agent_0"))
    critic_encoder = CentralStateEncoder(env.state_space)
    actor_batch = actor_encoder.encode(
        [observations[agent] for agent in env.possible_agents],
        device=device,
    )
    critic_batch = critic_encoder.encode([env.state()], device=device)
    actor = SharedActor(
        input_dim=actor_encoder.input_dim,
        move_action_count=actor_encoder.move_action_count,
        message_action_count=actor_encoder.message_action_count,
        config=_network_config(),
        seed=11,
    )
    critic = CentralizedCritic(
        input_dim=critic_encoder.input_dim,
        config=_network_config(),
        seed=12,
    )

    logits = actor(actor_batch.inputs)
    values = critic(critic_batch)
    assert logits.move.shape == (easy_config.num_agents, 5)
    assert logits.message.shape == (easy_config.num_agents, 5)
    assert values.shape == (1,)
    assert actor_batch.move_action_masks.shape == (easy_config.num_agents, 5)
    assert actor_batch.message_action_masks.shape == (easy_config.num_agents, 5)


def test_actor_encoder_rejects_centralized_state(easy_config: EnvConfig) -> None:
    env = GridRescueParallelEnv(easy_config)
    env.reset(seed=4)
    encoder = ActorObservationEncoder(env.observation_space("agent_0"))
    with pytest.raises(TrainingError, match="decentralized observation space"):
        encoder.encode([env.state()], device=torch.device("cpu"))


def test_actor_and_critic_runtime_input_types_are_separate() -> None:
    config = _network_config()
    actor = SharedActor(
        input_dim=3,
        move_action_count=5,
        message_action_count=2,
        config=config,
        seed=1,
    )
    critic = CentralizedCritic(input_dim=4, config=config, seed=2)
    with pytest.raises(TypeError, match="ActorInput only"):
        actor(cast(ActorInput, CriticInput(torch.zeros((1, 4)))))
    with pytest.raises(TypeError, match="CriticInput only"):
        critic(cast(CriticInput, ActorInput(torch.zeros((1, 3)))))


def test_network_initialization_is_explicitly_seeded_and_isolated() -> None:
    config = _network_config()
    torch.manual_seed(999)
    global_before = torch.random.get_rng_state().clone()
    first = SharedActor(
        input_dim=3,
        move_action_count=5,
        message_action_count=2,
        config=config,
        seed=7,
    )
    global_after = torch.random.get_rng_state()
    second = SharedActor(
        input_dim=3,
        move_action_count=5,
        message_action_count=2,
        config=config,
        seed=7,
    )
    assert torch.equal(global_before, global_after)
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        )
    )


def test_non_finite_network_inputs_are_rejected() -> None:
    actor = SharedActor(
        input_dim=2,
        move_action_count=2,
        message_action_count=2,
        config=_network_config(),
        seed=0,
    )
    with pytest.raises(NumericalError, match="NaN or infinite"):
        actor(ActorInput(torch.tensor([[float("nan"), 0.0]])))


def test_one_shared_actor_parameter_set_receives_all_agent_gradients() -> None:
    actor = SharedActor(
        input_dim=2,
        move_action_count=2,
        message_action_count=2,
        config=_network_config(),
        seed=0,
    )
    logits = actor(ActorInput(torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
    (logits.move.sum() + logits.message.sum()).backward()
    assert all(parameter.grad is not None for parameter in actor.parameters())


def test_torch_generators_are_explicit_and_repeatable() -> None:
    device = resolve_device("cpu")
    first = torch.rand(5, generator=make_torch_generator(19, device), device=device)
    second = torch.rand(5, generator=make_torch_generator(19, device), device=device)
    assert torch.equal(first, second)


def test_device_resolution_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    with pytest.raises(ConfigurationError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_deterministic_kernel_mode_is_explicitly_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        configure_deterministic_algorithms(True)
        assert torch.are_deterministic_algorithms_enabled()
        configure_deterministic_algorithms(False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous)
