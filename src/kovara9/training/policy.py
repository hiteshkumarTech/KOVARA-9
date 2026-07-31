"""Decentralized action selection for the one shared actor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kovara9.core.errors import TrainingError
from kovara9.training.distributions import (
    FactoredActionDistribution,
    FactoredActionStatistics,
)
from kovara9.training.encoding import EncodedActorBatch
from kovara9.training.networks import ActorInput, SharedActor


@dataclass(frozen=True, slots=True)
class PolicyActionSelection:
    """Local actor inputs, masks, and selected factored actions."""

    actor_input: ActorInput
    move_action_masks: Tensor
    message_action_masks: Tensor
    statistics: FactoredActionStatistics


def select_actions(
    actor: SharedActor,
    actor_batch: EncodedActorBatch,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> PolicyActionSelection:
    """Select actions without accepting or deriving any centralized state."""

    if not isinstance(actor_batch.inputs, ActorInput):
        raise TypeError("policy selection accepts ActorInput only")
    distribution = FactoredActionDistribution(
        actor(actor_batch.inputs),
        move_action_masks=actor_batch.move_action_masks,
        message_action_masks=actor_batch.message_action_masks,
    )
    if deterministic:
        statistics = distribution.mode()
    else:
        if generator is None:
            raise TrainingError("stochastic action selection requires an explicit generator")
        statistics = distribution.sample(generator=generator)
    return PolicyActionSelection(
        actor_input=actor_batch.inputs,
        move_action_masks=actor_batch.move_action_masks,
        message_action_masks=actor_batch.message_action_masks,
        statistics=statistics,
    )
