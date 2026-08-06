from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import load_bundled_demo_config
from kovara9.config.models import DemoConfig
from kovara9.core.errors import ArtifactError
from kovara9.core.types import WorldSnapshot
from kovara9.demo import run_demo, write_demo_artifacts
from kovara9.evaluation.runner import run_episode


def _short_demo() -> DemoConfig:
    config = load_bundled_demo_config()
    return config.model_copy(
        update={"environment": config.environment.model_copy(update={"max_steps": 3})}
    )


def test_bundled_demo_declares_distinct_training_domain_baselines() -> None:
    config = load_bundled_demo_config()
    assert {episode.policy for episode in config.episodes} == {"random", "frontier"}
    assert len({episode.seed for episode in config.episodes}) == len(config.episodes)
    assert all(
        episode.seed in config.seed_partitions.train.resolved_seeds for episode in config.episodes
    )
    assert all(
        episode.seed not in config.seed_partitions.test.resolved_seeds
        for episode in config.episodes
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate-name", "names must be unique"),
        ("duplicate-seed", "seeds must be unique"),
        ("missing-baseline", "both random and frontier"),
        ("non-training-seed", "outside the declared train partition"),
    ],
)
def test_demo_configuration_rejects_ambiguous_or_partition_unsafe_cases(
    mutation: str,
    message: str,
) -> None:
    payload = load_bundled_demo_config().model_dump(mode="python", round_trip=True)
    episodes = payload["episodes"]
    assert isinstance(episodes, tuple)
    first = dict(episodes[0])
    second = dict(episodes[1])
    if mutation == "duplicate-name":
        second["name"] = first["name"]
    elif mutation == "duplicate-seed":
        second["seed"] = first["seed"]
    elif mutation == "missing-baseline":
        second["policy"] = first["policy"]
    else:
        second["seed"] = 20_000
    payload["episodes"] = (first, second)
    with pytest.raises(ValidationError, match=message):
        DemoConfig.model_validate(payload)


def test_demo_execution_and_rendering_are_reproducible() -> None:
    config = _short_demo()
    first = run_demo(config)
    second = run_demo(config)
    without_frames = run_demo(config, capture_frames=False)
    assert first.to_dict() == second.to_dict()
    assert first.to_dict() == without_frames.to_dict()
    assert tuple(episode.frames for episode in first.episodes) == tuple(
        episode.frames for episode in second.episodes
    )
    random, frontier = first.episodes
    assert random.frames == ()
    assert len(frontier.frames) == frontier.record.episode_length + 1
    assert all(episode.record.seed == episode.specification.seed for episode in first.episodes)
    assert first.to_dict()["training_performed"] is False
    assert first.to_dict()["final_evaluation_performed"] is False


def test_demo_frame_capture_is_explicitly_bounded() -> None:
    config = _short_demo().model_copy(update={"frame_capture_limit": 1})
    run = run_demo(config)
    rendered = next(episode for episode in run.episodes if episode.specification.render)
    assert len(rendered.frames) == 1
    assert rendered.frames_truncated is True


def test_snapshot_observer_cannot_change_simulator_outcomes_or_metrics() -> None:
    config = _short_demo()

    def mutate_copy(snapshot: WorldSnapshot) -> None:
        snapshot.agent_positions.clear()
        snapshot.communication_budgets.clear()
        snapshot.latest_messages.clear()
        with pytest.raises(ValueError, match="read-only"):
            snapshot.obstacles[0, 0] = True

    expected = run_episode(
        env_config=config.environment,
        seed=4242,
        policy_factory=RandomPolicy,
    )
    observed = run_episode(
        env_config=config.environment,
        seed=4242,
        policy_factory=RandomPolicy,
        snapshot_observer=mutate_copy,
    )
    assert observed == expected


def test_demo_artifacts_are_deterministic_and_collision_safe(tmp_path: Path) -> None:
    config = _short_demo()
    run = run_demo(config, capture_frames=False)
    output = tmp_path / "demo"
    write_demo_artifacts(output, config, run)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report == run.to_dict()
    assert (output / "demo.resolved.yaml").is_file()
    with pytest.raises(ArtifactError, match="already exists"):
        write_demo_artifacts(output, config, run)
