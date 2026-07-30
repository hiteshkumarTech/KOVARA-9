"""Collision-safe, inspectable evaluation artifact persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from kovara9 import __version__
from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import ArtifactError
from kovara9.evaluation.runner import EvaluationResult


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _lock_hash(root: Path) -> str | None:
    lock_path = root / "uv.lock"
    try:
        content = lock_path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(content).hexdigest()


class ArtifactWriter:
    """Create a run directory exactly once and mark it complete last."""

    def __init__(self, output_dir: Path, project_root: Path | None = None) -> None:
        self.output_dir = output_dir
        self.project_root = project_root or Path.cwd()

    def write(  # noqa: PLR0913
        self,
        *,
        env_config: EnvConfig,
        evaluation_config: EvaluationConfig,
        result: EvaluationResult,
        held_out_env_config: EnvConfig | None = None,
        held_out_result: EvaluationResult | None = None,
        comparison: dict[str, Any] | None = None,
    ) -> None:
        """Persist all records; refuse ambiguous partial comparison inputs."""

        if (held_out_env_config is None) != (held_out_result is None):
            raise ArtifactError("held-out config and result must be provided together")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            self._write_yaml("environment.resolved.yaml", env_config.model_dump(mode="json"))
            self._write_yaml(
                "evaluation.resolved.yaml",
                evaluation_config.model_dump(mode="json", exclude_none=True),
            )
            self._write_jsonl("episodes.jsonl", [record.to_dict() for record in result.records])
            self._write_json("summary.json", result.summary.to_dict())
            if held_out_env_config is not None and held_out_result is not None:
                self._write_yaml(
                    "held_out_environment.resolved.yaml",
                    held_out_env_config.model_dump(mode="json"),
                )
                self._write_jsonl(
                    "held_out_episodes.jsonl",
                    [record.to_dict() for record in held_out_result.records],
                )
                self._write_json("held_out_summary.json", held_out_result.summary.to_dict())
            if comparison is not None:
                self._write_json("generalization.json", comparison)
            manifest = {
                "schema_version": 1,
                "status": "complete",
                "project": "kovara9",
                "project_version": __version__,
                "policy": result.summary.policy,
                "python": sys.version,
                "platform": platform.platform(),
                "git_revision": _git_revision(),
                "uv_lock_sha256": _lock_hash(self.project_root),
                "seeds": list(evaluation_config.resolved_seeds),
            }
            self._write_json("manifest.json", manifest)
        except FileExistsError as exc:
            raise ArtifactError(f"output directory already exists: {self.output_dir}") from exc
        except OSError as exc:
            raise ArtifactError(f"cannot write artifacts to {self.output_dir}: {exc}") from exc

    def _write_json(self, name: str, payload: Any) -> None:
        self._atomic_text(name, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _write_jsonl(self, name: str, records: list[dict[str, Any]]) -> None:
        text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        self._atomic_text(name, text)

    def _write_yaml(self, name: str, payload: Any) -> None:
        self._atomic_text(name, yaml.safe_dump(payload, sort_keys=True))

    def _atomic_text(self, name: str, text: str) -> None:
        destination = self.output_dir / name
        temporary = self.output_dir / f".{name}.tmp"
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(destination)
