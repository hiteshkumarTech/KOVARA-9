import json
import subprocess
from pathlib import Path

import pytest

from kovara9.config.models import (
    EnvConfig,
    EvaluationConfig,
    SeedPartitionsConfig,
    SeedRangeConfig,
)
from kovara9.core.errors import ArtifactError
from kovara9.evaluation.metrics import aggregate_records
from kovara9.evaluation.records import EpisodeRecord
from kovara9.evaluation.runner import EvaluationResult
from kovara9.reporting.artifacts import ArtifactWriter
from kovara9.reporting.summaries import comparison_summary

PARTITIONS = SeedPartitionsConfig(
    train=SeedRangeConfig(start=0, count=10_000),
    validation=SeedRangeConfig(start=10_000, count=1_000),
    test=SeedRangeConfig(start=20_000, count=1_000),
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _evaluation(name: str = "artifact") -> EvaluationConfig:
    return EvaluationConfig(
        name=name,
        seeds=(20000,),
        seed_partition="test",
        seed_partitions=PARTITIONS,
        bootstrap_samples=0,
    )


def _result(success: bool) -> EvaluationResult:
    record = EpisodeRecord(
        seed=20000,
        success=success,
        episode_length=4,
        targets_recovered=int(success),
        total_targets=1,
        exploration_coverage=0.5,
        duplicated_exploration=0.2,
        communication_messages=0,
        messages_per_agent_step=0,
        team_efficiency=0.1,
        shared_return=1,
        termination_reason="success" if success else "time_limit",
    )
    config = _evaluation()
    return EvaluationResult(
        records=(record,),
        summary=aggregate_records([record], config, "random"),
        policy_parameters={"message_probability": 0.1},
    )


def test_artifacts_are_complete_parseable_and_collision_safe(
    tmp_path: Path,
    easy_config: EnvConfig,
) -> None:
    output = tmp_path / "run"
    evaluation = _evaluation()
    result = _result(True)
    ArtifactWriter(output, project_root=PROJECT_ROOT).write(
        env_config=easy_config,
        evaluation_config=evaluation,
        result=result,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    episode = json.loads((output / "episodes.jsonl").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["policy"] == {
        "name": "random",
        "parameters": {"message_probability": 0.1},
    }
    assert manifest["configured_episode_seeds"] == [20000]
    assert manifest["executed_episode_seeds"] == [20000]
    assert manifest["configuration_fingerprints"]["environment"]
    assert manifest["uv_lock_sha256"]
    assert summary["metrics"]["success_rate"]["mean"] == 1
    assert episode["seed"] == 20000
    assert not list(output.glob("*.tmp"))
    with pytest.raises(ArtifactError, match="already exists"):
        ArtifactWriter(output).write(
            env_config=easy_config,
            evaluation_config=evaluation,
            result=result,
        )


def test_comparison_artifacts_and_input_pair_validation(
    tmp_path: Path,
    easy_config: EnvConfig,
) -> None:
    reference = _result(True)
    held_out = _result(False)
    held_out_config = easy_config.model_copy(update={"max_steps": easy_config.max_steps + 1})
    comparison = comparison_summary(reference, held_out, easy_config, held_out_config)
    evaluation = _evaluation("compare")
    output = tmp_path / "comparison"
    ArtifactWriter(output).write(
        env_config=easy_config,
        evaluation_config=evaluation,
        result=reference,
        held_out_env_config=held_out_config,
        held_out_result=held_out,
        comparison=comparison,
    )
    stored = json.loads((output / "generalization.json").read_text(encoding="utf-8"))
    assert stored["generalization_gap"] == 1
    assert (output / "held_out_episodes.jsonl").exists()
    with pytest.raises(ArtifactError, match="provided together"):
        ArtifactWriter(tmp_path / "invalid").write(
            env_config=easy_config,
            evaluation_config=evaluation,
            result=reference,
            held_out_env_config=easy_config,
        )


def test_artifacts_reject_result_seed_inconsistency(
    tmp_path: Path,
    easy_config: EnvConfig,
) -> None:
    valid = _result(True)
    inconsistent_record = EpisodeRecord(**{**valid.records[0].to_dict(), "seed": 20001})
    inconsistent = EvaluationResult(
        records=(inconsistent_record,),
        summary=valid.summary,
        policy_parameters=valid.policy_parameters,
    )
    with pytest.raises(ArtifactError, match="result seeds do not match"):
        ArtifactWriter(tmp_path / "bad-seeds").write(
            env_config=easy_config,
            evaluation_config=_evaluation(),
            result=inconsistent,
        )
    assert not (tmp_path / "bad-seeds").exists()


def test_git_provenance_discovers_root_from_subdirectory_and_dirty_tree(
    tmp_path: Path,
    easy_config: EnvConfig,
) -> None:
    repository = tmp_path / "repository"
    nested = repository / "nested" / "directory"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "KOVARA Test"],
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "test fixture"],
        check=True,
        capture_output=True,
    )
    tracked.write_text("dirty\n", encoding="utf-8")

    output = tmp_path / "dirty-run"
    ArtifactWriter(output, project_root=nested).write(
        env_config=easy_config,
        evaluation_config=_evaluation(),
        result=_result(True),
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["git"]["repository_root"]) == repository.resolve()
    assert manifest["git"]["commit"]
    assert manifest["git"]["dirty"] is True
