"""Safe YAML loading with contextual configuration errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import ConfigurationError


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

    return _load(path, EvaluationConfig)
