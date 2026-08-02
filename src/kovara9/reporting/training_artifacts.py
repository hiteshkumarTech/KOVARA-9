"""Atomic, inspectable artifacts for one local training invocation."""

from __future__ import annotations

import json
import platform
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from kovara9.config.loader import TrainingInputs, configuration_fingerprint
from kovara9.core.errors import ArtifactError
from kovara9.reporting.artifacts import _git_provenance, _lock_hash, _package_versions
from kovara9.training.checkpoint import training_definition_fingerprint
from kovara9.training.protocols import TrainingProgress

type TrainingArtifactStatus = Literal["running", "bounded", "complete"]


class TrainingArtifactWriter:
    """Own a collision-safe run directory and update each file atomically."""

    def __init__(self, output_dir: Path, project_root: Path | None = None) -> None:
        self.output_dir = output_dir
        self.project_root = project_root or Path.cwd()
        self._started = False

    def start(self, inputs: TrainingInputs, *, resume_from: Path | None) -> None:
        """Create resolved inputs and provenance before training mutates models."""

        if self._started:
            raise ArtifactError("training artifact writer has already started")
        git = _git_provenance(self.project_root)
        lock_root = (
            Path(git.repository_root) if git.repository_root else self.project_root.resolve()
        )
        provenance = {
            "schema_version": 1,
            "project": "kovara9",
            "git": {
                "repository_root": git.repository_root,
                "commit": git.commit,
                "dirty": git.dirty,
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "uv_lock_sha256": _lock_hash(lock_root),
            "resume_from": str(resume_from.resolve()) if resume_from is not None else None,
        }
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            (self.output_dir / "checkpoints").mkdir()
            self._atomic_yaml(
                "training.resolved.yaml",
                inputs.training.model_dump(mode="json", exclude_none=True),
            )
            self._atomic_yaml(
                "environment.resolved.yaml",
                inputs.environment.model_dump(mode="json", exclude_none=True),
            )
            self._atomic_yaml(
                "validation.resolved.yaml",
                inputs.validation.model_dump(mode="json", exclude_none=True),
            )
            self._atomic_json("provenance.json", provenance)
            self._atomic_json(
                "manifest.json",
                self._manifest(
                    inputs,
                    status="running",
                    progress=TrainingProgress(0, 0, 0),
                    latest_checkpoint=None,
                    best_checkpoint=None,
                    best_validation=None,
                    wall_clock_seconds=0.0,
                ),
            )
        except FileExistsError as exc:
            raise ArtifactError(f"output directory already exists: {self.output_dir}") from exc
        except OSError as exc:
            raise ArtifactError(
                f"cannot initialize training artifacts in {self.output_dir}: {exc}"
            ) from exc
        self._started = True

    def write_metrics(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Replace the complete metric journal in one atomic operation."""

        self._require_started()
        try:
            text = "".join(
                json.dumps(dict(record), allow_nan=False, sort_keys=True) + "\n"
                for record in records
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"training metrics are not JSON serializable: {exc}") from exc
        self._atomic_text("metrics.jsonl", text)

    def update_manifest(  # noqa: PLR0913
        self,
        inputs: TrainingInputs,
        *,
        status: TrainingArtifactStatus,
        progress: TrainingProgress,
        latest_checkpoint: Path,
        best_checkpoint: Path | None,
        best_validation: Mapping[str, float] | None,
        wall_clock_seconds: float,
    ) -> None:
        """Publish progress only after the referenced checkpoint is complete."""

        self._require_started()
        self._atomic_json(
            "manifest.json",
            self._manifest(
                inputs,
                status=status,
                progress=progress,
                latest_checkpoint=latest_checkpoint,
                best_checkpoint=best_checkpoint,
                best_validation=best_validation,
                wall_clock_seconds=wall_clock_seconds,
            ),
        )

    def checkpoint_path(self, environment_steps: int) -> Path:
        """Return the canonical immutable path for one scheduled checkpoint."""

        self._require_started()
        return self.output_dir / "checkpoints" / f"step-{environment_steps:012d}.pt"

    def publish_best_checkpoint(self, source: Path) -> Path:
        """Atomically copy an immutable validated checkpoint to the best alias."""

        self._require_started()
        destination = self.output_dir / "checkpoints" / "best.pt"
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ArtifactError(f"cannot publish best checkpoint {destination}: {exc}") from exc
        return destination

    def _manifest(  # noqa: PLR0913
        self,
        inputs: TrainingInputs,
        *,
        status: TrainingArtifactStatus,
        progress: TrainingProgress,
        latest_checkpoint: Path | None,
        best_checkpoint: Path | None,
        best_validation: Mapping[str, float] | None,
        wall_clock_seconds: float,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "status": status,
            "algorithm": inputs.training.algorithm,
            "seed": inputs.training.seed,
            "configuration_fingerprints": {
                "training": training_definition_fingerprint(inputs.training),
                "environment": configuration_fingerprint(inputs.environment),
                "validation": configuration_fingerprint(inputs.validation),
            },
            "progress": {
                "environment_steps": progress.environment_steps,
                "agent_transitions": (progress.environment_steps * inputs.environment.num_agents),
                "optimizer_updates": progress.optimizer_updates,
                "completed_episodes": progress.completed_episodes,
            },
            "latest_checkpoint": (
                str(latest_checkpoint.relative_to(self.output_dir))
                if latest_checkpoint is not None
                else None
            ),
            "best_checkpoint": (
                str(best_checkpoint.relative_to(self.output_dir))
                if best_checkpoint is not None
                else None
            ),
            "best_validation": dict(best_validation) if best_validation is not None else None,
            "wall_clock_seconds": wall_clock_seconds,
        }

    def _require_started(self) -> None:
        if not self._started:
            raise ArtifactError("training artifact writer has not started")

    def _atomic_json(self, name: str, payload: Any) -> None:
        try:
            text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"cannot serialize training artifact {name}: {exc}") from exc
        self._atomic_text(name, text)

    def _atomic_yaml(self, name: str, payload: Any) -> None:
        self._atomic_text(name, yaml.safe_dump(payload, sort_keys=True))

    def _atomic_text(self, name: str, text: str) -> None:
        destination = self.output_dir / name
        temporary = self.output_dir / f".{name}.tmp"
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(destination)
        except OSError as exc:
            raise ArtifactError(f"cannot write training artifact {destination}: {exc}") from exc
