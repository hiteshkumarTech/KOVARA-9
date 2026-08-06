"""Fast, packaged walkthrough built from real simulator transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.policy import Policy
from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import configuration_fingerprint
from kovara9.config.models import DemoConfig, DemoEpisodeConfig
from kovara9.core.errors import ArtifactError
from kovara9.core.types import WorldSnapshot
from kovara9.evaluation.records import EpisodeRecord
from kovara9.evaluation.runner import PolicyFactory, SnapshotObserver, run_episode
from kovara9.rendering.ansi import AnsiRenderer
from kovara9.rendering.protocol import Renderer


@dataclass(frozen=True, slots=True)
class DemoEpisodeRun:
    """One executed demo episode and any presentation-only rendered frames."""

    specification: DemoEpisodeConfig
    record: EpisodeRecord
    frames: tuple[str, ...]
    frames_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic episode evidence without presentation frames."""

        return {
            "name": self.specification.name,
            "policy": self.specification.policy,
            "render_declared": self.specification.render,
            "record": self.record.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DemoRun:
    """Complete non-benchmark walkthrough result."""

    name: str
    configuration_fingerprint: str
    episodes: tuple[DemoEpisodeRun, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, honest machine-readable report."""

        return {
            "schema_version": 1,
            "status": "complete",
            "name": self.name,
            "classification": "behavioral_walkthrough_not_benchmark",
            "configuration_fingerprint": self.configuration_fingerprint,
            "training_performed": False,
            "final_evaluation_performed": False,
            "learned_checkpoint_loaded": False,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def _policy_factory(specification: DemoEpisodeConfig) -> PolicyFactory:
    policy_type: type[Policy] = RandomPolicy if specification.policy == "random" else FrontierPolicy
    return policy_type


def _frame_observer(
    renderer: Renderer[str],
    frames: list[str],
    limit: int,
) -> SnapshotObserver:
    def observe(snapshot: WorldSnapshot) -> None:
        if len(frames) < limit:
            frames.append(renderer.render(snapshot))

    return observe


def run_demo(config: DemoConfig, *, capture_frames: bool = True) -> DemoRun:
    """Execute the configured baseline examples with explicit local seed streams."""

    config = DemoConfig.model_validate(config.model_dump(mode="python", round_trip=True))
    renderer = AnsiRenderer()
    executed: list[DemoEpisodeRun] = []
    for specification in config.episodes:
        frames: list[str] = []
        should_render = capture_frames and specification.render
        record = run_episode(
            env_config=config.environment,
            seed=specification.seed,
            policy_factory=_policy_factory(specification),
            snapshot_observer=(
                _frame_observer(renderer, frames, config.frame_capture_limit)
                if should_render
                else None
            ),
        )
        frame_count = record.episode_length + 1 if should_render else 0
        executed.append(
            DemoEpisodeRun(
                specification=specification,
                record=record,
                frames=tuple(frames),
                frames_truncated=len(frames) < frame_count,
            )
        )
    return DemoRun(
        name=config.name,
        configuration_fingerprint=configuration_fingerprint(config),
        episodes=tuple(executed),
    )


def write_demo_artifacts(output_directory: Path, config: DemoConfig, run: DemoRun) -> None:
    """Persist a resolved configuration and report in a new output directory."""

    try:
        output_directory.mkdir(parents=True, exist_ok=False)
        _write_text(
            output_directory / "demo.resolved.yaml",
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
        )
        _write_text(
            output_directory / "report.json",
            json.dumps(run.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        )
    except FileExistsError as exc:
        raise ArtifactError(f"demo output directory already exists: {output_directory}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot write demo artifacts to {output_directory}: {exc}") from exc


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
