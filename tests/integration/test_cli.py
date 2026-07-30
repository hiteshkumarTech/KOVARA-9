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


@pytest.mark.integration
def test_cli_episode_and_evaluation_artifacts(tmp_path: Path) -> None:
    environment = yaml.safe_load(
        Path("configs/environments/grid_rescue_easy.yaml").read_text(encoding="utf-8")
    )
    environment["max_steps"] = 2
    env_path = tmp_path / "env.yaml"
    env_path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    evaluation = {
        "schema_version": 1,
        "name": "cli-smoke",
        "seeds": [20000],
        "bootstrap_samples": 0,
        "bootstrap_confidence": 0.95,
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
def test_cli_surfaces_invalid_configuration(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not: an-environment\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", str(invalid)])
    assert result.exit_code == 2
    assert "command_failed" in result.stdout
