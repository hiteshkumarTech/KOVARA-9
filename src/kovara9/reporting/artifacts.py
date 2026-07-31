"""Collision-safe, inspectable evaluation artifact persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kovara9 import __version__
from kovara9.config.loader import configuration_fingerprint
from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import ArtifactError
from kovara9.evaluation.runner import EvaluationResult


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """Git identity captured before artifact files are created."""

    repository_root: str | None
    commit: str | None
    dirty: bool | None


def _git_provenance(start: Path) -> GitProvenance:
    location = start.resolve()
    if location.is_file():
        location = location.parent
    try:
        root_result = subprocess.run(
            ["git", "-C", str(location), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return GitProvenance(None, None, None)
    if root_result.returncode != 0 or not root_result.stdout.strip():
        return GitProvenance(None, None, None)
    repository_root = Path(root_result.stdout.strip()).resolve()
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_result = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ArtifactError(
            f"cannot determine Git provenance from repository {repository_root}: {exc}"
        ) from exc
    commit = commit_result.stdout.strip()
    if not commit:
        raise ArtifactError(f"Git returned an empty commit for repository {repository_root}")
    return GitProvenance(str(repository_root), commit, bool(status_result.stdout.strip()))


def _package_versions() -> dict[str, str]:
    distributions = (
        "gymnasium",
        "numpy",
        "pettingzoo",
        "pydantic",
        "PyYAML",
        "structlog",
        "typer",
    )
    versions = {"kovara9": __version__}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ArtifactError(
                f"cannot record package version because {distribution} is not installed"
            ) from exc
    return versions


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
        if (held_out_result is None) != (comparison is None):
            raise ArtifactError("comparison summary and held-out result must be provided together")
        configured_seeds = evaluation_config.resolved_seeds
        executed_seeds = tuple(record.seed for record in result.records)
        if executed_seeds != configured_seeds:
            raise ArtifactError(
                "reference result seeds do not match evaluation configuration: "
                f"configured={configured_seeds}, executed={executed_seeds}"
            )
        if held_out_result is not None:
            held_out_seeds = tuple(record.seed for record in held_out_result.records)
            if held_out_seeds != configured_seeds:
                raise ArtifactError(
                    "held-out result seeds do not match evaluation configuration: "
                    f"configured={configured_seeds}, executed={held_out_seeds}"
                )
            if held_out_result.summary.policy != result.summary.policy:
                raise ArtifactError("reference and held-out results use different policies")
            if dict(held_out_result.policy_parameters) != dict(result.policy_parameters):
                raise ArtifactError(
                    "reference and held-out results use different policy parameters"
                )

        environment_fingerprint = configuration_fingerprint(env_config)
        held_out_fingerprint = (
            configuration_fingerprint(held_out_env_config)
            if held_out_env_config is not None
            else None
        )
        if held_out_fingerprint == environment_fingerprint:
            raise ArtifactError(
                "cannot write a generalization comparison for semantically identical environments"
            )
        git = _git_provenance(self.project_root)
        lock_root = (
            Path(git.repository_root) if git.repository_root else self.project_root.resolve()
        )
        manifest = {
            "schema_version": 2,
            "status": "complete",
            "project": "kovara9",
            "git": {
                "repository_root": git.repository_root,
                "commit": git.commit,
                "dirty": git.dirty,
            },
            "configuration_fingerprints": {
                "environment": environment_fingerprint,
                "held_out_environment": held_out_fingerprint,
                "evaluation": configuration_fingerprint(evaluation_config),
            },
            "policy": {
                "name": result.summary.policy,
                "parameters": dict(result.policy_parameters),
            },
            "configured_episode_seeds": list(configured_seeds),
            "executed_episode_seeds": list(executed_seeds),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "uv_lock_sha256": _lock_hash(lock_root),
        }
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
