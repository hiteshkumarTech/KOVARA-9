import pytest
import torch

from kovara9.core.errors import NumericalError, TrainingError
from kovara9.training.config import NetworkConfig
from kovara9.training.distributions import FactoredActionDistribution
from kovara9.training.encoding import EncodedActorBatch
from kovara9.training.networks import ActorInput, ActorLogits, SharedActor
from kovara9.training.policy import select_actions
from kovara9.training.runtime import make_torch_generator


def _distribution() -> FactoredActionDistribution:
    return FactoredActionDistribution(
        ActorLogits(
            move=torch.tensor([[[1.0, 9.0, 2.0], [3.0, 2.0, 1.0]]]),
            message=torch.tensor([[[5.0, 1.0], [1.0, 4.0]]]),
        ),
        move_action_masks=torch.tensor([[[True, False, True], [True, True, True]]]),
        message_action_masks=torch.tensor([[[True, False], [True, True]]]),
    )


def _actor() -> SharedActor:
    return SharedActor(
        input_dim=3,
        move_action_count=3,
        message_action_count=2,
        config=NetworkConfig(
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="tanh",
        ),
        seed=4,
    )


def _batch() -> EncodedActorBatch:
    return EncodedActorBatch(
        inputs=ActorInput(torch.tensor([[0.25, 0.5, 0.75], [1.0, 0.0, 0.5]])),
        move_action_masks=torch.tensor([[True, False, True], [True, True, False]]),
        message_action_masks=torch.tensor([[True, False], [True, True]]),
    )


def test_masked_probabilities_are_normalized_and_invalid_entries_are_zero() -> None:
    probabilities = _distribution().probabilities
    assert probabilities.move[0, 0, 1].item() == 0.0
    assert probabilities.message[0, 0, 1].item() == 0.0
    assert torch.allclose(probabilities.move.sum(dim=-1), torch.ones((1, 2)))
    assert torch.allclose(probabilities.message.sum(dim=-1), torch.ones((1, 2)))


@pytest.mark.parametrize(
    ("move_mask", "message_mask", "match"),
    [
        (torch.zeros((1, 3), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool), "move"),
        (torch.ones((1, 3), dtype=torch.bool), torch.zeros((1, 2), dtype=torch.bool), "message"),
    ],
)
def test_all_invalid_masks_are_rejected(
    move_mask: torch.Tensor,
    message_mask: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(TrainingError, match=match):
        FactoredActionDistribution(
            ActorLogits(move=torch.zeros((1, 3)), message=torch.zeros((1, 2))),
            move_action_masks=move_mask,
            message_action_masks=message_mask,
        )


def test_mask_shape_dtype_and_non_finite_logits_are_validated() -> None:
    logits = ActorLogits(move=torch.zeros((2, 3)), message=torch.zeros((2, 2)))
    with pytest.raises(TrainingError, match="shape"):
        FactoredActionDistribution(
            logits,
            move_action_masks=torch.ones((2, 2), dtype=torch.bool),
            message_action_masks=torch.ones((2, 2), dtype=torch.bool),
        )
    with pytest.raises(TrainingError, match="bool"):
        FactoredActionDistribution(
            logits,
            move_action_masks=torch.ones((2, 3)),
            message_action_masks=torch.ones((2, 2), dtype=torch.bool),
        )
    with pytest.raises(NumericalError, match="move logits"):
        FactoredActionDistribution(
            ActorLogits(move=torch.tensor([[float("nan"), 0.0]]), message=torch.zeros((1, 2))),
            move_action_masks=torch.ones((1, 2), dtype=torch.bool),
            message_action_masks=torch.ones((1, 2), dtype=torch.bool),
        )


def test_mode_is_masked_argmax_and_joint_statistics_are_factor_sums() -> None:
    statistics = _distribution().mode()
    assert statistics.move_actions.tolist() == [[2, 0]]
    assert statistics.message_actions.tolist() == [[0, 1]]
    assert torch.equal(
        statistics.joint_log_probabilities,
        statistics.move_log_probabilities + statistics.message_log_probabilities,
    )
    assert torch.equal(
        statistics.joint_entropy,
        statistics.move_entropy + statistics.message_entropy,
    )


def test_invalid_supplied_actions_are_rejected() -> None:
    distribution = _distribution()
    with pytest.raises(TrainingError, match="rejected by the mask"):
        distribution.evaluate_actions(
            torch.tensor([[1, 0]], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
        )
    with pytest.raises(TrainingError, match="int64"):
        distribution.evaluate_actions(
            torch.tensor([[2.0, 0.0]]),
            torch.tensor([[0, 1]], dtype=torch.int64),
        )


def test_deterministic_policy_selection_is_repeatable() -> None:
    first = select_actions(_actor(), _batch(), deterministic=True)
    second = select_actions(_actor(), _batch(), deterministic=True)
    assert torch.equal(first.statistics.move_actions, second.statistics.move_actions)
    assert torch.equal(first.statistics.message_actions, second.statistics.message_actions)


def test_stochastic_policy_selection_uses_only_explicit_generator() -> None:
    actor = _actor()
    device = torch.device("cpu")
    first = select_actions(
        actor,
        _batch(),
        deterministic=False,
        generator=make_torch_generator(22, device),
    )
    second = select_actions(
        actor,
        _batch(),
        deterministic=False,
        generator=make_torch_generator(22, device),
    )
    assert torch.equal(first.statistics.move_actions, second.statistics.move_actions)
    assert torch.equal(first.statistics.message_actions, second.statistics.message_actions)
    with pytest.raises(TrainingError, match="explicit generator"):
        select_actions(actor, _batch(), deterministic=False)


def test_different_sampling_seeds_can_select_different_actions() -> None:
    actor = _actor()
    batch = EncodedActorBatch(
        inputs=ActorInput(torch.zeros((64, 3))),
        move_action_masks=torch.ones((64, 3), dtype=torch.bool),
        message_action_masks=torch.ones((64, 2), dtype=torch.bool),
    )
    first = select_actions(
        actor,
        batch,
        deterministic=False,
        generator=make_torch_generator(1, torch.device("cpu")),
    )
    second = select_actions(
        actor,
        batch,
        deterministic=False,
        generator=make_torch_generator(2, torch.device("cpu")),
    )
    assert not torch.equal(first.statistics.move_actions, second.statistics.move_actions)
