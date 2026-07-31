import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from kovara9.cli import app

runner = CliRunner()


@pytest.mark.integration
def test_cli_help_and_validation() -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "KOVARA-9" in help_result.stdout
    valid = runner.invoke(
        app,
        ["config", "validate", "configs/environments/grid_rescue_easy.yaml"],
    )
    assert valid.exit_code == 0
    assert "configuration_valid" in valid.stdout
    training = runner.invoke(
        app,
        ["config", "validate", "configs/training/mappo_smoke.yaml"],
    )
    assert training.exit_code == 0
    assert "kind=training" in training.stdout


@pytest.mark.integration
def test_cli_episode_and_evaluation_artifacts(tmp_path: Path) -> None:
    environment = yaml.safe_load(
        Path("configs/environments/grid_rescue_easy.yaml").read_text(encoding="utf-8")
    )
    environment["max_steps"] = 2
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    evaluation = {
        "schema_version": 2,
        "name": "cli-smoke",
        "seeds": [20000],
        "bootstrap_samples": 0,
        "bootstrap_confidence": 0.95,
        "seed_partition": "test",
        "seed_partitions": {
            "train": {"start": 0, "count": 10_000},
            "validation": {"start": 10_000, "count": 1_000},
            "test": {"start": 20_000, "count": 1_000},
        },
    }
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(evaluation), encoding="utf-8")

    episode = runner.invoke(
        app,
        [
            "--json-logs",
            "env",
            "run",
            "--config",
            str(env_path),
            "--agent",
            "random",
            "--seed",
            "3",
            "--render",
            "none",
        ],
    )
    assert episode.exit_code == 0
    assert '"event": "episode_complete"' in episode.stdout

    output = tmp_path / "artifacts"
    evaluated = runner.invoke(
        app,
        [
            "--json-logs",
            "evaluate",
            "--env-config",
            str(env_path),
            "--eval-config",
            str(eval_path),
            "--agent",
            "random",
            "--output",
            str(output),
        ],
    )
    assert evaluated.exit_code == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"


@pytest.mark.integration
def test_cli_generalization_uses_only_authoritative_config_paths(tmp_path: Path) -> None:
    reference = yaml.safe_load(
        Path("configs/environments/grid_rescue_easy.yaml").read_text(encoding="utf-8")
    )
    reference["max_steps"] = 1
    held_out = {**reference, "max_steps": 2}
    reference_path = tmp_path / "reference.yaml"
    held_out_path = tmp_path / "held-out.yaml"
    reference_path.write_text(yaml.safe_dump(reference), encoding="utf-8")
    held_out_path.write_text(yaml.safe_dump(held_out), encoding="utf-8")
    evaluation = {
        "schema_version": 2,
        "name": "generalization-smoke",
        "seeds": [20000],
        "bootstrap_samples": 0,
        "bootstrap_confidence": 0.95,
        "seed_partition": "test",
        "seed_partitions": {
            "train": {"start": 0, "count": 10_000},
            "validation": {"start": 10_000, "count": 1_000},
            "test": {"start": 20_000, "count": 1_000},
        },
        "comparison": {
            "reference_environment": reference_path.name,
            "held_out_environment": held_out_path.name,
        },
    }
    evaluation_path = tmp_path / "generalization.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation), encoding="utf-8")

    output = tmp_path / "generalization"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--eval-config",
            str(evaluation_path),
            "--agent",
            "random",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    comparison = json.loads((output / "generalization.json").read_text(encoding="utf-8"))
    assert comparison["comparison_type"] == "structural_generalization"
    assert (
        comparison["reference_environment_fingerprint"]
        != comparison["held_out_environment_fingerprint"]
    )

    conflict = runner.invoke(
        app,
        [
            "evaluate",
            "--eval-config",
            str(evaluation_path),
            "--env-config",
            str(reference_path),
            "--output",
            str(tmp_path / "conflict"),
        ],
    )
    assert conflict.exit_code == 2
    assert "must be omitted" in conflict.stdout


@pytest.mark.integration
def test_cli_surfaces_invalid_configuration(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not: an-environment\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", str(invalid)])
    assert result.exit_code == 2
    assert "command_failed" in result.stdout
