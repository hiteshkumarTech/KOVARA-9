import json
from pathlib import Path

import pytest

from kovara9.config.models import EnvConfig, EvaluationConfig
from kovara9.core.errors import ArtifactError
from kovara9.evaluation.metrics import aggregate_records
from kovara9.evaluation.records import EpisodeRecord
from kovara9.evaluation.runner import EvaluationResult
from kovara9.reporting.artifacts import ArtifactWriter
from kovara9.reporting.summaries import comparison_summary


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
    config = EvaluationConfig(name="artifact", seeds=(20000,), bootstrap_samples=0)
    return EvaluationResult(
        records=(record,),
        summary=aggregate_records([record], config, "random"),
    )


def test_artifacts_are_complete_parseable_and_collision_safe(
    tmp_path: Path,
    easy_config: EnvConfig,
) -> None:
    output = tmp_path / "run"
    evaluation = EvaluationConfig(name="artifact", seeds=(20000,), bootstrap_samples=0)
    result = _result(True)
    ArtifactWriter(output, project_root=tmp_path).write(
        env_config=easy_config,
        evaluation_config=evaluation,
        result=result,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    episode = json.loads((output / "episodes.jsonl").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["uv_lock_sha256"] is None
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
    comparison = comparison_summary(reference, held_out)
    evaluation = EvaluationConfig(name="compare", seeds=(20000,), bootstrap_samples=0)
    output = tmp_path / "comparison"
    ArtifactWriter(output).write(
        env_config=easy_config,
        evaluation_config=evaluation,
        result=reference,
        held_out_env_config=easy_config,
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
