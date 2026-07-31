import math
from dataclasses import replace

import pytest
import torch

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.config import NetworkConfig
from kovara9.training.distributions import FactoredActionDistribution
from kovara9.training.losses import PPOMinibatch, clipped_surrogate_loss, mappo_style_loss
from kovara9.training.networks import (
    ActorInput,
    ActorLogits,
    CentralizedCritic,
    CriticInput,
    SharedActor,
)


def _networks() -> tuple[SharedActor, CentralizedCritic]:
    config = NetworkConfig(
        actor_hidden_sizes=(8,),
        critic_hidden_sizes=(8,),
        activation="tanh",
    )
    return (
        SharedActor(
            input_dim=3,
            move_action_count=2,
            message_action_count=2,
            config=config,
            seed=1,
        ),
        CentralizedCritic(input_dim=4, config=config, seed=2),
    )


def _minibatch(actor: SharedActor, critic: CentralizedCritic) -> PPOMinibatch:
    actor_features = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
    move_masks = torch.ones((2, 2), dtype=torch.bool)
    message_masks = torch.ones((2, 2), dtype=torch.bool)
    move_actions = torch.tensor([0, 1], dtype=torch.int64)
    message_actions = torch.tensor([1, 0], dtype=torch.int64)
    with torch.no_grad():
        statistics = FactoredActionDistribution(
            actor(ActorInput(actor_features)),
            move_action_masks=move_masks,
            message_action_masks=message_masks,
        ).evaluate_actions(move_actions, message_actions)
    return PPOMinibatch(
        actor_features=actor_features,
        critic_features=torch.tensor([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.5, -0.5]]),
        move_action_masks=move_masks,
        message_action_masks=message_masks,
        move_actions=move_actions,
        message_actions=message_actions,
        old_joint_log_probabilities=statistics.joint_log_probabilities,
        advantages=torch.tensor([1.0, -0.5]),
        value_targets=torch.tensor([0.5, -0.25]),
        valid_samples=torch.ones(2, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_loss"),
    [
        (1.5, 1.0, -1.2),
        (0.5, -1.0, 0.8),
        (1.1, 2.0, -2.2),
    ],
)
def test_clipped_surrogate_above_below_and_inside_range(
    ratio: float,
    advantage: float,
    expected_loss: float,
) -> None:
    diagnostics = clipped_surrogate_loss(
        new_log_probabilities=torch.tensor([math.log(ratio)]),
        old_log_probabilities=torch.zeros(1),
        advantages=torch.tensor([advantage]),
        valid_samples=torch.ones(1, dtype=torch.bool),
        clipping_coefficient=0.2,
    )
    assert diagnostics.probability_ratios.item() == pytest.approx(ratio)
    assert diagnostics.policy_loss.item() == pytest.approx(expected_loss)


def test_surrogate_diagnostics_match_controlled_ratios() -> None:
    ratios = torch.tensor([1.0, 1.5])
    diagnostics = clipped_surrogate_loss(
        new_log_probabilities=torch.log(ratios),
        old_log_probabilities=torch.zeros(2),
        advantages=torch.ones(2),
        valid_samples=torch.ones(2, dtype=torch.bool),
        clipping_coefficient=0.2,
    )
    expected_kl = (((ratios - 1.0) - torch.log(ratios)).mean()).item()
    assert diagnostics.approximate_kl.item() == pytest.approx(expected_kl)
    assert diagnostics.clip_fraction.item() == pytest.approx(0.5)
    assert diagnostics.mean_probability_ratio.item() == pytest.approx(1.25)


def test_invalid_samples_do_not_affect_policy_loss() -> None:
    diagnostics = clipped_surrogate_loss(
        new_log_probabilities=torch.tensor([0.0, math.log(100.0)]),
        old_log_probabilities=torch.zeros(2),
        advantages=torch.tensor([2.0, -1_000.0]),
        valid_samples=torch.tensor([True, False]),
        clipping_coefficient=0.2,
    )
    assert diagnostics.policy_loss.item() == pytest.approx(-2.0)
    assert diagnostics.valid_sample_count == 1


def test_mappo_loss_uses_joint_factored_log_probability() -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    diagnostics = mappo_style_loss(
        actor=actor,
        critic=critic,
        batch=batch,
        clipping_coefficient=0.2,
        entropy_coefficient=0.0,
        value_coefficient=0.0,
        explained_variance_epsilon=1e-8,
    )
    assert torch.allclose(diagnostics.probability_ratios, torch.ones(2))
    assert diagnostics.policy_loss.item() == pytest.approx(-batch.advantages.mean().item())
    assert torch.allclose(
        diagnostics.entropy,
        diagnostics.move_entropy + diagnostics.message_entropy,
    )


def test_centralized_critic_predictions_define_value_loss() -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    diagnostics = mappo_style_loss(
        actor=actor,
        critic=critic,
        batch=batch,
        clipping_coefficient=0.2,
        entropy_coefficient=0.0,
        value_coefficient=1.0,
        explained_variance_epsilon=1e-8,
    )
    expected = 0.5 * torch.square(batch.value_targets).mean()
    assert torch.allclose(diagnostics.value_loss, expected)


def test_entropy_and_value_coefficients_change_total_loss_exactly() -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    base = mappo_style_loss(
        actor=actor,
        critic=critic,
        batch=batch,
        clipping_coefficient=0.2,
        entropy_coefficient=0.0,
        value_coefficient=0.0,
        explained_variance_epsilon=1e-8,
    )
    weighted = mappo_style_loss(
        actor=actor,
        critic=critic,
        batch=batch,
        clipping_coefficient=0.2,
        entropy_coefficient=0.03,
        value_coefficient=0.7,
        explained_variance_epsilon=1e-8,
    )
    expected = base.total_loss + 0.7 * weighted.value_loss - 0.03 * weighted.entropy
    assert torch.allclose(weighted.total_loss, expected)


def test_actor_and_centralized_critic_receive_only_their_typed_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    actor_inputs: list[ActorInput] = []
    critic_inputs: list[CriticInput] = []
    original_actor = actor.forward
    original_critic = critic.forward

    def actor_forward(inputs: ActorInput) -> ActorLogits:
        actor_inputs.append(inputs)
        return original_actor(inputs)

    def critic_forward(inputs: CriticInput) -> torch.Tensor:
        critic_inputs.append(inputs)
        return original_critic(inputs)

    monkeypatch.setattr(actor, "forward", actor_forward)
    monkeypatch.setattr(critic, "forward", critic_forward)
    mappo_style_loss(
        actor=actor,
        critic=critic,
        batch=batch,
        clipping_coefficient=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        explained_variance_epsilon=1e-8,
    )
    assert len(actor_inputs) == 1
    assert len(critic_inputs) == 1
    assert actor_inputs[0].features.shape == (2, 3)
    assert critic_inputs[0].features.shape == (2, 4)


def test_loss_rejects_empty_or_non_finite_samples() -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    with pytest.raises(TrainingError, match="zero valid"):
        mappo_style_loss(
            actor=actor,
            critic=critic,
            batch=replace(batch, valid_samples=torch.zeros(2, dtype=torch.bool)),
            clipping_coefficient=0.2,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            explained_variance_epsilon=1e-8,
        )
    with pytest.raises(NumericalError, match="advantages"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.zeros(1),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.tensor([float("nan")]),
            valid_samples=torch.ones(1, dtype=torch.bool),
            clipping_coefficient=0.2,
        )


def test_surrogate_rejects_invalid_shapes_dtypes_and_empty_masks() -> None:
    with pytest.raises(ValueError, match="clipping_coefficient"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.zeros(1),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(1),
            valid_samples=torch.ones(1, dtype=torch.bool),
            clipping_coefficient=0.0,
        )
    with pytest.raises(TrainingError, match="mismatch"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.zeros(2),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(2),
            valid_samples=torch.ones(2, dtype=torch.bool),
            clipping_coefficient=0.2,
        )
    with pytest.raises(TrainingError, match="bool"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.zeros(1),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(1),
            valid_samples=torch.ones(1),
            clipping_coefficient=0.2,
        )
    with pytest.raises(TrainingError, match="at least one"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.zeros(1),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(1),
            valid_samples=torch.zeros(1, dtype=torch.bool),
            clipping_coefficient=0.2,
        )
    with pytest.raises(TrainingError, match="floating-point"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.zeros(1),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(1, dtype=torch.int64),
            valid_samples=torch.ones(1, dtype=torch.bool),
            clipping_coefficient=0.2,
        )
    with pytest.raises(NumericalError, match="probability ratios"):
        clipped_surrogate_loss(
            new_log_probabilities=torch.tensor([1_000.0]),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(1),
            valid_samples=torch.ones(1, dtype=torch.bool),
            clipping_coefficient=0.2,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("actor_features", torch.zeros((2, 4)), "actor_features"),
        ("critic_features", torch.zeros((2, 3)), "critic_features"),
        ("move_action_masks", torch.ones((2, 3), dtype=torch.bool), "movement masks"),
        ("message_action_masks", torch.ones((2, 3), dtype=torch.bool), "message masks"),
        ("advantages", torch.ones(1), "advantages must have shape"),
        ("move_action_masks", torch.ones((2, 2)), "move_action_masks must use bool"),
        ("move_actions", torch.zeros(2), "move_actions must use int64"),
    ],
)
def test_mappo_loss_rejects_incompatible_minibatch_contracts(
    field: str,
    value: torch.Tensor,
    match: str,
) -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    with pytest.raises(TrainingError, match=match):
        mappo_style_loss(
            actor=actor,
            critic=critic,
            batch=replace(batch, **{field: value}),
            clipping_coefficient=0.2,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            explained_variance_epsilon=1e-8,
        )


def test_mappo_loss_rejects_stale_action_masks_and_invalid_coefficients() -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    stale_move_mask = batch.move_action_masks.clone()
    stale_move_mask[0] = torch.tensor([False, True])
    with pytest.raises(TrainingError, match="rejected by the mask"):
        mappo_style_loss(
            actor=actor,
            critic=critic,
            batch=replace(batch, move_action_masks=stale_move_mask),
            clipping_coefficient=0.2,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            explained_variance_epsilon=1e-8,
        )
    for coefficients, match in (
        ({"entropy_coefficient": -0.1}, "entropy_coefficient"),
        ({"value_coefficient": -0.1}, "value_coefficient"),
        ({"explained_variance_epsilon": 0.0}, "explained_variance_epsilon"),
    ):
        arguments = {
            "actor": actor,
            "critic": critic,
            "batch": batch,
            "clipping_coefficient": 0.2,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "explained_variance_epsilon": 1e-8,
        }
        arguments.update(coefficients)
        with pytest.raises(ValueError, match=match):
            mappo_style_loss(**arguments)


def test_minibatch_selection_rejects_non_vector_integer_indices() -> None:
    actor, critic = _networks()
    batch = _minibatch(actor, critic)
    with pytest.raises(TrainingError, match="one-dimensional int64"):
        batch.select(torch.zeros((1, 1), dtype=torch.int64))
    with pytest.raises(TrainingError, match="one-dimensional int64"):
        batch.select(torch.zeros(1))
