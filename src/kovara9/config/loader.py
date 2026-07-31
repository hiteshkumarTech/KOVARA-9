"""Safe YAML loading with contextual configuration errors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from kovara9.config.models import ComparisonConfig, EnvConfig, EvaluationConfig
from kovara9.core.errors import ConfigurationError
from kovara9.training.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class TrainingInputs:
    """Validated training configuration and its referenced experiment inputs."""

    training: TrainingConfig
    environment: EnvConfig
    validation: EvaluationConfig


def _load[ConfigT: BaseModel](path: Path, model_type: type[ConfigT]) -> ConfigT:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
    try:
        raw: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration {path} must contain a YAML mapping")
    try:
        return model_type.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration {path}:\n{exc}") from exc


def load_environment_config(path: Path) -> EnvConfig:
    """Load and validate an environment configuration."""

    return _load(path, EnvConfig)


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load and validate an evaluation configuration."""

    config = _load(path, EvaluationConfig)
    if config.comparison is None:
        return config
    base_directory = path.resolve().parent
    reference_path = _resolve_config_path(
        config.comparison.reference_environment,
        base_directory,
    )
    held_out_path = _resolve_config_path(
        config.comparison.held_out_environment,
        base_directory,
    )
    if reference_path == held_out_path:
        raise ConfigurationError(
            "reference and held-out environment paths resolve to the same file"
        )
    comparison = ComparisonConfig(
        reference_environment=reference_path,
        held_out_environment=held_out_path,
    )
    return config.model_copy(update={"comparison": comparison})


def _resolve_config_path(path: Path, base_directory: Path) -> Path:
    """Resolve a declared path relative to its owning configuration file."""

    candidate = path if path.is_absolute() else base_directory / path
    return candidate.resolve()


def configuration_fingerprint(config: BaseModel) -> str:
    """Hash one fully validated configuration using a canonical JSON encoding."""

    payload = json.dumps(
        config.model_dump(mode="json", exclude_none=False),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_comparison_environment_configs(
    evaluation_config: EvaluationConfig,
) -> tuple[EnvConfig, EnvConfig]:
    """Load an authoritative, semantically distinct environment comparison."""

    comparison = evaluation_config.comparison
    if comparison is None:
        raise ConfigurationError("evaluation configuration does not declare a comparison")
    reference = load_environment_config(comparison.reference_environment)
    held_out = load_environment_config(comparison.held_out_environment)
    if configuration_fingerprint(reference) == configuration_fingerprint(held_out):
        raise ConfigurationError(
            "reference and held-out environment configurations are semantically identical"
        )
    return reference, held_out


def load_training_config(path: Path) -> TrainingConfig:
    """Load training configuration and resolve owned paths portably."""

    config = _load(path, TrainingConfig)
    base_directory = path.resolve().parent
    return config.model_copy(
        update={
            "environment_config": _resolve_config_path(
                config.environment_config,
                base_directory,
            ),
            "validation_config": _resolve_config_path(
                config.validation_config,
                base_directory,
            ),
        }
    )


def load_training_inputs(path: Path) -> TrainingInputs:
    """Load and cross-validate one complete training experiment definition."""

    training = load_training_config(path)
    environment = load_environment_config(training.environment_config)
    validation = load_evaluation_config(training.validation_config)
    if validation.comparison is not None:
        raise ConfigurationError("training validation configuration cannot declare a comparison")
    if validation.seed_partition != "validation":
        raise ConfigurationError(
            "training validation configuration must select the validation seed partition"
        )
    if training.seed not in validation.seed_partitions.train.resolved_seeds:
        raise ConfigurationError(
            f"training seed {training.seed} is outside the declared train seed partition"
        )
    rollout_transitions = training.rollout_environment_steps * environment.num_agents
    if training.minibatch_size > rollout_transitions:
        raise ConfigurationError(
            "minibatch_size cannot exceed rollout_length * num_environments * num_agents"
        )
    if rollout_transitions % training.minibatch_size != 0:
        raise ConfigurationError("rollout agent transitions must be divisible by minibatch_size")
    return TrainingInputs(training, environment, validation)
