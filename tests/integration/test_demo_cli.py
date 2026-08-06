from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from kovara9.cli import app
from kovara9.config.loader import load_bundled_demo_config

runner = CliRunner()


@pytest.mark.integration
def test_packaged_demo_validates_without_repository_inputs() -> None:
    result = runner.invoke(app, ["demo", "--validate-only"])
    assert result.exit_code == 0, result.stdout
    assert "open_source_demo_valid" in result.stdout
    assert "training_performed=False" in result.stdout
    assert "final_evaluation_performed=False" in result.stdout


@pytest.mark.integration
def test_cli_runs_custom_demo_and_writes_transparent_artifacts(tmp_path: Path) -> None:
    config = load_bundled_demo_config()
    short_config = config.model_copy(
        update={"environment": config.environment.model_copy(update={"max_steps": 2})}
    )
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        yaml.safe_dump(short_config.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    validated = runner.invoke(app, ["config", "validate", str(config_path)])
    assert validated.exit_code == 0, validated.stdout
    assert "kind=demo" in validated.stdout

    result = runner.invoke(
        app,
        ["demo", "--config", str(config_path), "--no-render", "--output", str(output)],
    )
    assert result.exit_code == 0, result.stdout
    assert "not benchmark estimates" in result.stdout
    assert "open_source_demo_complete" in result.stdout
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["classification"] == "behavioral_walkthrough_not_benchmark"
    assert report["training_performed"] is False
    assert report["final_evaluation_performed"] is False
    assert [episode["record"]["seed"] for episode in report["episodes"]] == [4242, 4243]

    collision = runner.invoke(
        app,
        ["demo", "--config", str(config_path), "--no-render", "--output", str(output)],
    )
    assert collision.exit_code == 2
    assert "already exists" in collision.stdout
