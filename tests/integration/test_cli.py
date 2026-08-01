import json
import math
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from kovara9.cli import app
from kovara9.training.checkpoint import load_training_checkpoint, model_state_sha256

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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("config", "steps", "environment_count", "agent_count"),
    [
        ("configs/training/mappo_smoke.yaml", 3, 1, 2),
        ("configs/training/mappo_small.yaml", 2, 2, 3),
    ],
)
def test_cli_rollout_smoke_reports_untrained_tensor_collection(
    config: str,
    steps: int,
    environment_count: int,
    agent_count: int,
) -> None:
    result = runner.invoke(
        app,
        [
            "--json-logs",
            "rollout-smoke",
            "--training-config",
            config,
            "--steps",
            str(steps),
        ],
    )
    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout.strip())
    assert record["event"] == "rollout_smoke_complete"
    assert record["training_performed"] is False
    assert record["benchmark"] is False
    assert record["seed"] == 0
    assert record["device"] == "cpu"
    assert record["actor_shape"][:3] == [steps, environment_count, agent_count]
    assert record["critic_shape"][:2] == [steps, environment_count]
    assert record["transition_count"] == steps * environment_count * agent_count


@pytest.mark.integration
def test_cli_update_smoke_changes_finite_actor_and_critic_parameters() -> None:
    result = runner.invoke(
        app,
        [
            "--json-logs",
            "update-smoke",
            "--training-config",
            "configs/training/mappo_smoke.yaml",
        ],
    )
    assert result.exit_code == 0, result.stdout
    record = json.loads(result.stdout.strip())
    assert record["event"] == "optimization_smoke_complete"
    assert record["optimization_smoke_test"] is True
    assert record["benchmark"] is False
    assert record["useful_policy_learned"] is False
    assert record["full_training_run"] is False
    assert record["checkpoint_saved"] is False
    assert record["actor_parameters_changed"] is True
    assert record["critic_parameters_changed"] is True
    assert record["parameters_finite"] is True
    assert record["device"] == "cpu"
    assert record["valid_sample_count"] == 16
    assert record["minibatch_count"] == 2
    assert record["maximum_post_clip_gradient_norm"] <= 0.5


@pytest.mark.integration
def test_cli_training_checkpoint_evaluation_and_policy_comparison(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    environment = yaml.safe_load(
        Path("configs/environments/grid_rescue_easy.yaml").read_text(encoding="utf-8")
    )
    environment["max_steps"] = 2
    environment_path = tmp_path / "environment.yaml"
    environment_path.write_text(yaml.safe_dump(environment), encoding="utf-8")
    validation = {
        "schema_version": 2,
        "name": "day4-cli-validation",
        "seeds": [10_000],
        "bootstrap_samples": 0,
        "bootstrap_confidence": 0.95,
        "seed_partition": "validation",
        "seed_partitions": {
            "train": {"start": 0, "count": 10_000},
            "validation": {"start": 10_000, "count": 1_000},
            "test": {"start": 20_000, "count": 1_000},
        },
    }
    validation_path = tmp_path / "validation.yaml"
    validation_path.write_text(yaml.safe_dump(validation), encoding="utf-8")
    training = yaml.safe_load(Path("configs/training/mappo_smoke.yaml").read_text(encoding="utf-8"))
    training.update(
        {
            "environment_config": environment_path.name,
            "validation_config": validation_path.name,
            "rollout_length": 2,
            "ppo_epochs": 1,
            "minibatch_size": 4,
            "total_environment_steps": 4,
            "checkpoint_frequency": 2,
            "evaluation_frequency": 4,
        }
    )
    training_path = tmp_path / "training.yaml"
    training_path.write_text(yaml.safe_dump(training), encoding="utf-8")

    initial_output = tmp_path / "initial-run"
    initialized = runner.invoke(
        app,
        [
            "--json-logs",
            "train",
            "--training-config",
            str(training_path),
            "--output",
            str(initial_output),
            "--initialize-only",
        ],
    )
    assert initialized.exit_code == 0, initialized.stdout
    assert json.loads(initialized.stdout.strip())["event"] == "training_initialized"
    initial_manifest = json.loads((initial_output / "manifest.json").read_text(encoding="utf-8"))
    initial_checkpoint = initial_output / initial_manifest["latest_checkpoint"]
    initial_actor_identity = model_state_sha256(
        load_training_checkpoint(initial_checkpoint).actor_state
    )

    training_output = tmp_path / "training-run"
    trained = runner.invoke(
        app,
        [
            "--json-logs",
            "train",
            "--training-config",
            str(training_path),
            "--output",
            str(training_output),
        ],
    )
    assert trained.exit_code == 0, trained.stdout
    training_record = json.loads(trained.stdout.strip())
    assert training_record["event"] == "training_complete"
    manifest = json.loads((training_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    checkpoint = training_output / manifest["latest_checkpoint"]
    assert checkpoint.exists()
    assert (training_output / manifest["best_checkpoint"]).exists()
    assert manifest["progress"]["agent_transitions"] == 8
    assert manifest["wall_clock_seconds"] > 0.0
    update_records = [
        json.loads(line)
        for line in (training_output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert update_records
    assert all(
        record["accepted_communication_count"] + record["rejected_communication_count"]
        == record["selected_communication_count"]
        for record in update_records
    )
    assert all(math.isfinite(record["mean_reward"]) for record in update_records)

    evaluation_output = tmp_path / "checkpoint-evaluation"
    evaluated = runner.invoke(
        app,
        [
            "evaluate-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--env-config",
            str(environment_path),
            "--eval-config",
            str(validation_path),
            "--output",
            str(evaluation_output),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.stdout
    evaluation_manifest = json.loads(
        (evaluation_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert evaluation_manifest["policy"]["name"] == "checkpoint-shared-actor"
    assert evaluation_manifest["inference_performance"]["call_count"] > 0

    comparison_output = tmp_path / "policy-comparison"
    compared = runner.invoke(
        app,
        [
            "compare-policies",
            "--checkpoint",
            str(checkpoint),
            "--env-config",
            str(environment_path),
            "--eval-config",
            str(validation_path),
            "--output",
            str(comparison_output),
        ],
    )
    assert compared.exit_code == 0, compared.stdout
    comparison = json.loads(
        (comparison_output / "policy-comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["status"] == "complete"
    assert set(comparison["policies"]) == {"random", "frontier", "untrained", "checkpoint"}
    assert comparison["paired_episode_seeds"] == [10_000]
    assert comparison["paired_results"][0]["seed"] == 10_000
    assert set(comparison["paired_results"][0]["policies"]) == {
        "random",
        "frontier",
        "untrained",
        "checkpoint",
    }
    assert (
        comparison["policy_parameters"]["untrained"]["actor_state_sha256"] == initial_actor_identity
    )
    assert comparison["inference_performance"]["checkpoint"]["call_count"] > 0

    validation["name"] = "forbidden-test-comparison"
    validation["seeds"] = [20_000]
    validation["seed_partition"] = "test"
    test_validation_path = tmp_path / "test-validation.yaml"
    test_validation_path.write_text(yaml.safe_dump(validation), encoding="utf-8")
    forbidden = runner.invoke(
        app,
        [
            "compare-policies",
            "--checkpoint",
            str(checkpoint),
            "--env-config",
            str(environment_path),
            "--eval-config",
            str(test_validation_path),
            "--output",
            str(tmp_path / "forbidden-test-output"),
        ],
    )
    assert forbidden.exit_code == 2
    assert "refuses test seeds" in forbidden.stdout


@pytest.mark.integration
def test_day6_seed_command_preserves_root_identity_and_rejects_test_partition(
    tmp_path: Path,
) -> None:
    initialized = runner.invoke(
        app,
        [
            "--json-logs",
            "day6-run-seed",
            "--training-config",
            "configs/training/mappo_smoke.yaml",
            "--root-seed",
            "2",
            "--output",
            str(tmp_path / "day6-initial"),
            "--initialize-only",
        ],
    )
    assert initialized.exit_code == 0, initialized.stdout
    record = json.loads(initialized.stdout.strip())
    assert record["event"] == "day6_seed_initialized"
    assert record["root_seed"] == 2
    assert record["test_partition_consumed"] is False
    manifest = json.loads((tmp_path / "day6-initial" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2

    raw = yaml.safe_load(Path("configs/training/mappo_smoke.yaml").read_text(encoding="utf-8"))
    raw["environment_config"] = str(Path("configs/environments/grid_rescue_easy.yaml").resolve())
    raw["validation_config"] = str(Path("configs/evaluation/smoke.yaml").resolve())
    forbidden_path = tmp_path / "forbidden-day6.yaml"
    forbidden_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    forbidden = runner.invoke(
        app,
        [
            "day6-run-seed",
            "--training-config",
            str(forbidden_path),
            "--root-seed",
            "0",
            "--output",
            str(tmp_path / "forbidden-output"),
            "--initialize-only",
        ],
    )
    assert forbidden.exit_code == 2
    assert "validation seed partition" in forbidden.stdout
