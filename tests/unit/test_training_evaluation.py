from __future__ import annotations

from typing import Any

import pytest
import torch
from gymnasium import spaces
from gymnasium.spaces import Space

from kovara9.config.models import EnvConfig
from kovara9.core.errors import TrainingError
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv
from kovara9.training.config import NetworkConfig
from kovara9.training.encoding import ActorObservationEncoder
from kovara9.training.evaluation import DecentralizedActorPolicy
from kovara9.training.networks import SharedActor


def _actor_and_spaces(
    environment_config: EnvConfig,
    *,
    input_delta: int = 0,
    move_delta: int = 0,
    message_delta: int = 0,
) -> tuple[SharedActor, Space[Any], Space[Any]]:
    environment = GridRescueParallelEnv(environment_config)
    agent = environment.possible_agents[0]
    observation_space = environment.observation_space(agent)
    action_space = environment.action_space(agent)
    encoder = ActorObservationEncoder(observation_space)
    environment.close()
    actor = SharedActor(
        input_dim=encoder.input_dim + input_delta,
        move_action_count=encoder.move_action_count + move_delta,
        message_action_count=encoder.message_action_count + message_delta,
        config=NetworkConfig(
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="tanh",
        ),
        seed=1,
    )
    return actor, observation_space, action_space


def _policy(actor: SharedActor) -> DecentralizedActorPolicy:
    return DecentralizedActorPolicy(
        actor=actor,
        device=torch.device("cpu"),
        policy_name="test-actor",
        parameters={"deterministic": True},
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("agent", "non-empty agent"),
        ("seed", "seed must be non-negative"),
        ("dict", "action space must be a Dict"),
        ("discrete", "requires discrete"),
        ("input", "actor input"),
        ("move_head", "movement head"),
        ("message_head", "message head"),
        ("move_space", "movement action space"),
        ("message_space", "message action space"),
    ],
)
def test_checkpoint_policy_rejects_incompatible_local_contracts(
    easy_config: EnvConfig,
    case: str,
    message: str,
) -> None:
    actor, observation_space, action_space = _actor_and_spaces(
        easy_config,
        input_delta=int(case == "input"),
        move_delta=int(case == "move_head"),
        message_delta=int(case == "message_head"),
    )
    if case == "dict":
        action_space = spaces.Discrete(2)
    elif case == "discrete":
        action_space = spaces.Dict({"move": spaces.MultiBinary(5), "message": spaces.Discrete(5)})
    elif case == "move_space":
        action_space = spaces.Dict({"move": spaces.Discrete(6), "message": spaces.Discrete(5)})
    elif case == "message_space":
        action_space = spaces.Dict({"move": spaces.Discrete(5), "message": spaces.Discrete(6)})
    with pytest.raises(TrainingError, match=message):
        _policy(actor).reset(
            agent_id="" if case == "agent" else "agent_0",
            observation_space=observation_space,
            action_space=action_space,
            seed=-1 if case == "seed" else 0,
        )


def test_checkpoint_policy_requires_reset_before_action(easy_config: EnvConfig) -> None:
    actor, _observation_space, _action_space = _actor_and_spaces(easy_config)
    with pytest.raises(TrainingError, match="reset before acting"):
        _policy(actor).act({})
