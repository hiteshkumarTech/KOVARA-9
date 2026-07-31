from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import kovara9.training.optimization as optimization_module
from kovara9.config.models import EnvConfig
from kovara9.core.errors import NumericalError, TrainingError
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.training.collector import SynchronousRolloutCollector
from kovara9.training.config import NetworkConfig, TrainingConfig
from kovara9.training.encoding import ActorObservationEncoder, CentralStateEncoder
from kovara9.training.gae import GAEResult, compute_gae
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.optimization import PPOOptimizer, build_optimization_minibatch
from kovara9.training.rollout import RolloutBatch
from kovara9.training.seeds import ExperimentSeedStreams


def _training_config(
    *,
    ppo_epochs: int = 2,
    minibatch_size: int = 3,
    maximum_gradient_norm: float = 0.5,
) -> TrainingConfig:
    return TrainingConfig(
        environment_config=Path("environment.yaml"),
        validation_config=Path("validation.yaml"),
        network=NetworkConfig(
            actor_hidden_sizes=(16,),
            critic_hidden_sizes=(16,),
            activation="tanh",
        ),
        rollout_length=4,
        num_environments=1,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        discount_factor=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        advantage_normalization_epsilon=1e-8,
        clipping_coefficient=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        maximum_gradient_norm=maximum_gradient_norm,
        learning_rate=0.001,
        total_environment_steps=8,
        checkpoint_frequency=4,
        evaluation_frequency=4,
        device="cpu",
        deterministic_torch=True,
        seed=0,
    )


def _optimization_case(
    environment_config: EnvConfig,
    *,
    root_seed: int = 17,
    ppo_epochs: int = 2,
    minibatch_size: int = 3,
    maximum_gradient_norm: float = 0.5,
) -> tuple[SharedActor, CentralizedCritic, TrainingConfig, RolloutBatch, GAEResult]:
    config = _training_config(
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        maximum_gradient_norm=maximum_gradient_norm,
    )
    probe = GridRescueParallelEnv(environment_config)
    actor_encoder = ActorObservationEncoder(probe.observation_space(probe.possible_agents[0]))
    critic_encoder = CentralStateEncoder(probe.state_space)
    probe.close()
    streams = ExperimentSeedStreams(root_seed)
    actor = SharedActor(
        input_dim=actor_encoder.input_dim,
        move_action_count=actor_encoder.move_action_count,
        message_action_count=actor_encoder.message_action_count,
        config=config.network,
        seed=streams.actor_initialization,
    )
    critic = CentralizedCritic(
        input_dim=critic_encoder.input_dim,
        config=config.network,
        seed=streams.critic_initialization,
    )
    collector = SynchronousRolloutCollector(
        environment_factory=lambda: GridRescueParallelEnv(environment_config),
        num_environments=1,
        rollout_length=config.rollout_length,
        actor=actor,
        critic=critic,
        root_seed=root_seed,
        device=torch.device("cpu"),
    )
    try:
        batch = collector.collect().batch
    finally:
        collector.close()
    gae = compute_gae(
        batch,
        gamma=config.discount_factor,
        gae_lambda=config.gae_lambda,
        normalize_advantages=config.normalize_advantages,
        normalization_epsilon=config.advantage_normalization_epsilon,
    )
    return actor, critic, config, batch, gae


def _state(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _changed(before: tuple[torch.Tensor, ...], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(previous, current)
        for previous, current in zip(before, module.parameters(), strict=True)
    )


def test_build_optimization_batch_preserves_actor_and_centralized_critic_inputs(
    easy_config: EnvConfig,
) -> None:
    _actor, _critic, _config, rollout, gae = _optimization_case(easy_config)
    batch = build_optimization_minibatch(rollout, gae)
    assert batch.sample_count == int(rollout.active_agents.sum().item())
    assert batch.actor_features.shape[-1] == rollout.actor_features.shape[-1]
    assert batch.critic_features.shape[-1] == rollout.critic_features.shape[-1]
    assert torch.equal(batch.critic_features[0], batch.critic_features[1])
    assert bool(batch.valid_samples.all())


def test_equal_shuffle_seeds_produce_equal_orders_and_parameter_updates(
    easy_config: EnvConfig,
) -> None:
    first = _optimization_case(easy_config)
    second = _optimization_case(easy_config)
    first_optimizer = PPOOptimizer(
        actor=first[0], critic=first[1], config=first[2], shuffle_seed=111
    )
    second_optimizer = PPOOptimizer(
        actor=second[0], critic=second[1], config=second[2], shuffle_seed=111
    )
    first_result = first_optimizer.update(first[3], first[4])
    second_result = second_optimizer.update(second[3], second[4])
    assert first_result.epoch_sample_orders == second_result.epoch_sample_orders
    assert all(
        torch.equal(left, right)
        for left, right in zip(first[0].parameters(), second[0].parameters(), strict=True)
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(first[1].parameters(), second[1].parameters(), strict=True)
    )


def test_different_shuffle_seeds_change_minibatch_order(easy_config: EnvConfig) -> None:
    first = _optimization_case(easy_config, ppo_epochs=1)
    second = _optimization_case(easy_config, ppo_epochs=1)
    first_order = (
        PPOOptimizer(actor=first[0], critic=first[1], config=first[2], shuffle_seed=1)
        .update(first[3], first[4])
        .epoch_sample_orders
    )
    second_order = (
        PPOOptimizer(actor=second[0], critic=second[1], config=second[2], shuffle_seed=2)
        .update(second[3], second[4])
        .epoch_sample_orders
    )
    assert first_order != second_order


def test_uneven_minibatch_updates_actor_and_critic(easy_config: EnvConfig) -> None:
    actor, critic, config, rollout, gae = _optimization_case(
        easy_config, ppo_epochs=1, minibatch_size=3
    )
    actor_before = _state(actor)
    critic_before = _state(critic)
    result = PPOOptimizer(
        actor=actor,
        critic=critic,
        config=config,
        shuffle_seed=ExperimentSeedStreams(17).optimizer_shuffle,
    ).update(rollout, gae)
    assert result.valid_sample_count == 8
    assert result.minibatch_count == 3
    assert _changed(actor_before, actor)
    assert _changed(critic_before, critic)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in actor.parameters())
    assert all(bool(torch.isfinite(parameter).all()) for parameter in critic.parameters())


def test_gradient_clipping_is_enforced(easy_config: EnvConfig) -> None:
    actor, critic, config, rollout, gae = _optimization_case(
        easy_config,
        ppo_epochs=1,
        minibatch_size=8,
        maximum_gradient_norm=0.001,
    )
    result = PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=3).update(
        rollout, gae
    )
    assert result.maximum_pre_clip_gradient_norm > config.maximum_gradient_norm
    assert result.maximum_post_clip_gradient_norm == pytest.approx(
        config.maximum_gradient_norm,
        rel=1e-3,
    )


def test_optimizer_rejects_duplicate_parameters() -> None:
    config = _training_config()
    actor = SharedActor(
        input_dim=3,
        move_action_count=2,
        message_action_count=2,
        config=config.network,
        seed=1,
    )
    critic = CentralizedCritic(input_dim=3, config=config.network, seed=2)
    critic.encoder[0].weight = actor.encoder[0].weight
    with pytest.raises(TrainingError, match="duplicate"):
        PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=3)


def test_optimizer_revalidates_corrupted_configuration() -> None:
    config = _training_config()
    object.__setattr__(config, "learning_rate", -1.0)
    actor = SharedActor(
        input_dim=3,
        move_action_count=2,
        message_action_count=2,
        config=config.network,
        seed=1,
    )
    critic = CentralizedCritic(input_dim=4, config=config.network, seed=2)
    with pytest.raises(TrainingError, match="unvalidated training configuration"):
        PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=3)


@pytest.mark.parametrize("field", ["advantages", "value_targets"])
def test_optimizer_rejects_non_finite_advantage_or_return(
    easy_config: EnvConfig,
    field: str,
) -> None:
    actor, critic, config, rollout, gae = _optimization_case(easy_config)
    invalid = getattr(gae, field).clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(NumericalError, match=field.replace("_", " ")):
        PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1).update(
            rollout,
            replace(gae, **{field: invalid}),
        )


def test_optimizer_rejects_non_finite_loss(
    easy_config: EnvConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic, config, rollout, gae = _optimization_case(easy_config)
    original = optimization_module.mappo_style_loss

    def non_finite_loss(**kwargs: object) -> optimization_module.PPOLossDiagnostics:
        result = original(**kwargs)
        return replace(result, total_loss=torch.tensor(float("nan"), requires_grad=True))

    monkeypatch.setattr(optimization_module, "mappo_style_loss", non_finite_loss)
    with pytest.raises(NumericalError, match="total loss"):
        PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1).update(rollout, gae)


def test_optimizer_rejects_non_finite_gradient(easy_config: EnvConfig) -> None:
    actor, critic, config, rollout, gae = _optimization_case(easy_config)
    first_parameter = next(actor.parameters())
    hook = first_parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    try:
        with pytest.raises(NumericalError, match="gradient"):
            PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1).update(
                rollout, gae
            )
    finally:
        hook.remove()


@pytest.mark.parametrize("feature_name", ["actor_features", "critic_features"])
def test_optimizer_rejects_non_finite_model_inputs(
    easy_config: EnvConfig,
    feature_name: str,
) -> None:
    actor, critic, config, rollout, gae = _optimization_case(easy_config)
    invalid = getattr(rollout, feature_name).clone()
    invalid.reshape(-1)[0] = float("inf")
    feature_kind = feature_name.split("_", maxsplit=1)[0]
    with pytest.raises(NumericalError, match=feature_kind + ".*features"):
        PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1).update(
            replace(rollout, **{feature_name: invalid}), gae
        )


def test_optimizer_rejects_incompatible_rollout_and_gae_shapes(easy_config: EnvConfig) -> None:
    actor, critic, config, rollout, gae = _optimization_case(easy_config)
    optimizer = PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1)
    with pytest.raises(TrainingError, match="critic rollout features"):
        optimizer.update(replace(rollout, critic_features=rollout.critic_features[0]), gae)
    with pytest.raises(TrainingError, match="move_actions"):
        optimizer.update(replace(rollout, move_actions=rollout.move_actions[0]), gae)
    with pytest.raises(TrainingError, match="move_action_masks"):
        optimizer.update(
            replace(rollout, move_action_masks=rollout.move_action_masks[0]),
            gae,
        )
    with pytest.raises(TrainingError, match="GAE advantages"):
        optimizer.update(rollout, replace(gae, advantages=gae.advantages[0]))
    mismatched_valid = gae.valid_samples.clone()
    mismatched_valid[0, 0, 0] = False
    with pytest.raises(TrainingError, match="incompatible with rollout"):
        optimizer.update(rollout, replace(gae, valid_samples=mismatched_valid))


def test_optimizer_rejects_non_finite_initial_parameters(easy_config: EnvConfig) -> None:
    actor, critic, config, _rollout, _gae = _optimization_case(easy_config)
    with torch.no_grad():
        next(actor.parameters()).reshape(-1)[0] = float("nan")
    with pytest.raises(NumericalError, match="non-finite model parameters"):
        PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1)


def test_optimizer_rejects_empty_and_inconsistent_batches(easy_config: EnvConfig) -> None:
    actor, critic, config, rollout, gae = _optimization_case(easy_config)
    empty_rollout = replace(rollout, active_agents=torch.zeros_like(rollout.active_agents))
    empty_gae = replace(gae, valid_samples=torch.zeros_like(gae.valid_samples))
    optimizer = PPOOptimizer(actor=actor, critic=critic, config=config, shuffle_seed=1)
    with pytest.raises(TrainingError, match="at least one valid"):
        optimizer.update(empty_rollout, empty_gae)
    with pytest.raises(TrainingError, match="actor rollout features"):
        optimizer.update(replace(rollout, actor_features=rollout.actor_features[0]), gae)
