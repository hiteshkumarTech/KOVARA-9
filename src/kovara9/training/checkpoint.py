"""Atomic, validated persistence for reproducible training and evaluation."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import Field, ValidationError
from torch import Tensor

from kovara9.config.models import StrictModel
from kovara9.core.errors import TrainingError
from kovara9.training.config import TrainingConfig
from kovara9.training.networks import CentralizedCritic, SharedActor
from kovara9.training.protocols import TrainingProgress

_CHECKPOINT_SCHEMA_VERSION = 1
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class LearnerSignature(StrictModel):
    """Tensor and action contract that a checkpoint was built against."""

    actor_input_dim: int = Field(gt=0)
    critic_input_dim: int = Field(gt=0)
    move_action_count: int = Field(gt=0)
    message_action_count: int = Field(gt=0)
    agent_order: tuple[str, ...] = Field(min_length=1)


class CheckpointProgress(StrictModel):
    """Validated serialized training counters."""

    environment_steps: int = Field(ge=0)
    optimizer_updates: int = Field(ge=0)
    completed_episodes: int = Field(ge=0)

    @classmethod
    def from_progress(cls, progress: TrainingProgress) -> CheckpointProgress:
        return cls(
            environment_steps=progress.environment_steps,
            optimizer_updates=progress.optimizer_updates,
            completed_episodes=progress.completed_episodes,
        )

    def to_progress(self) -> TrainingProgress:
        return TrainingProgress(
            environment_steps=self.environment_steps,
            optimizer_updates=self.optimizer_updates,
            completed_episodes=self.completed_episodes,
        )


class CheckpointMetadata(StrictModel):
    """Inspectable configuration and compatibility metadata."""

    schema_version: int = Field(
        default=_CHECKPOINT_SCHEMA_VERSION,
        ge=_CHECKPOINT_SCHEMA_VERSION,
        le=_CHECKPOINT_SCHEMA_VERSION,
    )
    training_config: TrainingConfig
    training_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    environment_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    validation_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    signature: LearnerSignature
    progress: CheckpointProgress
    training_complete: bool


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated checkpoint payload, still independent from concrete modules."""

    metadata: CheckpointMetadata
    actor_state: Mapping[str, Tensor]
    critic_state: Mapping[str, Tensor]
    optimizer_state: Mapping[str, Any]
    collector_state: Mapping[str, Any]
    training_records: tuple[dict[str, Any], ...]


def training_definition_fingerprint(config: TrainingConfig) -> str:
    """Hash research parameters while excluding portable owned file locations."""

    payload = config.model_dump(
        mode="json",
        exclude={"environment_config", "validation_config"},
        exclude_none=False,
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_sha256(path: Path) -> str:
    """Return a stable identity for one complete checkpoint file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrainingError(f"cannot hash checkpoint {path}: {exc}") from exc
    return digest.hexdigest()


def save_training_checkpoint(  # noqa: PLR0913
    path: Path,
    *,
    metadata: CheckpointMetadata,
    actor: SharedActor,
    critic: CentralizedCritic,
    optimizer_state: Mapping[str, Any],
    collector_state: Mapping[str, Any],
    training_records: tuple[dict[str, Any], ...],
) -> None:
    """Atomically write one safe-load-compatible training checkpoint."""

    payload = {
        "metadata": metadata.model_dump(mode="json"),
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer_state": dict(optimizer_state),
        "collector_state": dict(collector_state),
        "training_records": list(training_records),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as exc:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise TrainingError(f"cannot write checkpoint {path}: {exc}") from exc


def load_training_checkpoint(path: Path) -> LoadedCheckpoint:
    """Load primitives and tensors only, then validate every outer contract."""

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise TrainingError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingError(f"checkpoint {path} must contain a mapping")
    expected_keys = {
        "metadata",
        "actor_state",
        "critic_state",
        "optimizer_state",
        "collector_state",
        "training_records",
    }
    if set(raw) != expected_keys:
        raise TrainingError(f"checkpoint {path} fields do not match schema version 1")
    try:
        metadata = CheckpointMetadata.model_validate(raw["metadata"])
    except ValidationError as exc:
        raise TrainingError(f"invalid checkpoint metadata in {path}: {exc}") from exc
    actor_state = _tensor_state_dict(path, "actor_state", raw["actor_state"])
    critic_state = _tensor_state_dict(path, "critic_state", raw["critic_state"])
    optimizer_state = raw["optimizer_state"]
    collector_state = raw["collector_state"]
    records = raw["training_records"]
    if not isinstance(optimizer_state, Mapping):
        raise TrainingError(f"checkpoint {path} optimizer state must be a mapping")
    if not isinstance(collector_state, Mapping):
        raise TrainingError(f"checkpoint {path} collector state must be a mapping")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise TrainingError(f"checkpoint {path} training records must be a list of mappings")
    return LoadedCheckpoint(
        metadata=metadata,
        actor_state=actor_state,
        critic_state=critic_state,
        optimizer_state=optimizer_state,
        collector_state=collector_state,
        training_records=tuple(dict(record) for record in records),
    )


def restore_model_states(
    checkpoint: LoadedCheckpoint,
    *,
    actor: SharedActor,
    critic: CentralizedCritic,
) -> None:
    """Strictly restore both networks and reject incompatible tensor layouts."""

    try:
        actor.load_state_dict(checkpoint.actor_state, strict=True)
        critic.load_state_dict(checkpoint.critic_state, strict=True)
    except RuntimeError as exc:
        raise TrainingError("checkpoint model tensors do not match learner signature") from exc
    for name, module in (("actor", actor), ("critic", critic)):
        if not all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters()):
            raise TrainingError(f"checkpoint restored non-finite {name} parameters")


def restore_actor_state(checkpoint: LoadedCheckpoint, *, actor: SharedActor) -> None:
    """Restore only the decentralized actor for inference."""

    try:
        actor.load_state_dict(checkpoint.actor_state, strict=True)
    except RuntimeError as exc:
        raise TrainingError("checkpoint actor tensors do not match observation signature") from exc
    if not all(bool(torch.isfinite(parameter).all()) for parameter in actor.parameters()):
        raise TrainingError("checkpoint restored non-finite actor parameters")


def _tensor_state_dict(path: Path, name: str, raw: Any) -> dict[str, Tensor]:
    if not isinstance(raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in raw.items()
    ):
        raise TrainingError(f"checkpoint {path} {name} must map names to tensors")
    return dict(raw)
