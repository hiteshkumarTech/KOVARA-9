from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kovara9.config.loader import load_training_config, load_training_inputs
from kovara9.core.errors import ConfigurationError
from kovara9.training.config import NetworkConfig, TrainingConfig


def _training_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "algorithm": "shared-actor-centralized-critic-ppo",
        "environment_config": "environment.yaml",
        "validation_config": "validation.yaml",
        "network": {
            "actor_hidden_sizes": [32],
            "critic_hidden_sizes": [32],
            "activation": "tanh",
        },
        "rollout_length": 4,
        "num_environments": 1,
        "ppo_epochs": 2,
        "minibatch_size": 4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "normalize_advantages": True,
        "advantage_normalization_epsilon": 1e-8,
        "clipping_coefficient": 0.2,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
        "maximum_gradient_norm": 0.5,
        "learning_rate": 0.0003,
        "total_environment_steps": 8,
        "checkpoint_frequency": 4,
        "evaluation_frequency": 4,
        "device": "cpu",
        "deterministic_torch": True,
        "seed": 0,
    }


def test_all_committed_training_configs_resolve_and_cross_validate() -> None:
    for path in Path("configs/training").glob("*.yaml"):
        inputs = load_training_inputs(path)
        assert inputs.training.environment_config.is_absolute()
        assert inputs.training.validation_config.is_absolute()
        assert inputs.validation.seed_partition == "validation"
        assert inputs.training.seed in inputs.validation.seed_partitions.train.resolved_seeds


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_environment_steps", 7, "total_environment_steps must be divisible"),
        ("checkpoint_frequency", 9, "cannot exceed"),
        ("evaluation_frequency", 3, "must be divisible"),
    ],
)
def test_training_schedule_rejects_inconsistent_frequencies(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _training_payload()
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        TrainingConfig.model_validate(payload)


def test_network_dimensions_are_bounded() -> None:
    with pytest.raises(ValidationError):
        NetworkConfig(actor_hidden_sizes=(0,), critic_hidden_sizes=(32,))


def test_training_paths_are_owned_by_the_training_file(tmp_path: Path) -> None:
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(_training_payload()), encoding="utf-8")
    config = load_training_config(path)
    assert config.environment_config == (tmp_path / "environment.yaml").resolve()
    assert config.validation_config == (tmp_path / "validation.yaml").resolve()


def test_training_seed_must_belong_to_declared_train_partition(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/training/mappo_smoke.yaml").read_text(encoding="utf-8"))
    payload["seed"] = 20_000
    payload["environment_config"] = str(
        Path("configs/environments/grid_rescue_easy.yaml").resolve()
    )
    payload["validation_config"] = str(
        Path("configs/evaluation/training_validation_smoke.yaml").resolve()
    )
    path = tmp_path / "invalid-seed.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="outside the declared train"):
        load_training_inputs(path)
