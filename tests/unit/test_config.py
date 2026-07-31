from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kovara9.config.loader import (
    load_comparison_environment_configs,
    load_environment_config,
    load_evaluation_config,
)
from kovara9.config.models import (
    CommunicationConfig,
    EnvConfig,
    RewardConfig,
    SeedPartitionsConfig,
    SeedRangeConfig,
)
from kovara9.core.errors import ConfigurationError


def test_all_committed_configs_load() -> None:
    for path in Path("configs/environments").glob("*.yaml"):
        assert load_environment_config(path).environment_id == "KovaraGridRescue-v0"
    for path in Path("configs/evaluation").glob("*.yaml"):
        assert load_evaluation_config(path).resolved_seeds


def test_evaluation_range_is_resolved() -> None:
    config = load_evaluation_config(Path("configs/evaluation/benchmark.yaml"))
    assert config.resolved_seeds[0] == 20000
    assert config.resolved_seeds[-1] == 20099
    assert len(config.resolved_seeds) == 100


def test_generalization_paths_resolve_relative_to_the_evaluation_file() -> None:
    config = load_evaluation_config(Path("configs/evaluation/generalization.yaml"))
    assert config.comparison is not None
    assert (
        config.comparison.reference_environment
        == Path("configs/environments/grid_rescue_medium.yaml").resolve()
    )
    assert (
        config.comparison.held_out_environment
        == Path("configs/environments/grid_rescue_hard.yaml").resolve()
    )
    reference, held_out = load_comparison_environment_configs(config)
    assert reference.num_agents == 3
    assert held_out.num_agents == 4


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "width: [",
        "width: 8\nheight: 8\nnum_agents: 2\nobstacle_density: 0.1\n"
        "observation_radius: 2\nnum_targets: 2\nmax_steps: 10\nunknown: true\n",
    ],
)
def test_bad_environment_files_raise_contextual_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"configuration|YAML"):
        load_environment_config(path)


def test_missing_file_is_contextual(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_environment_config(tmp_path / "missing.yaml")


def test_environment_capacity_and_id_are_validated(easy_config: EnvConfig) -> None:
    with pytest.raises(ValidationError, match="unsupported environment_id"):
        EnvConfig.model_validate({**easy_config.model_dump(), "environment_id": "Unknown-v0"})
    with pytest.raises(ValidationError, match="free cells"):
        EnvConfig.model_validate(
            {
                **easy_config.model_dump(),
                "width": 5,
                "height": 5,
                "num_agents": 4,
                "num_targets": 20,
                "obstacle_density": 0.45,
            }
        )


def test_disabled_communication_requires_zero_budget() -> None:
    with pytest.raises(ValidationError, match="must be 0"):
        CommunicationConfig(enabled=False, budget_per_agent=1)


@pytest.mark.parametrize(
    "field",
    ["target_recovery", "success_bonus", "step_penalty", "message_penalty"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rewards_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        RewardConfig.model_validate({field: value})


def test_model_copy_updates_are_revalidated(easy_config: EnvConfig) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 5"):
        easy_config.model_copy(update={"width": 1})


def test_seed_partition_overlap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="seed partitions overlap"):
        SeedPartitionsConfig(
            train=SeedRangeConfig(start=0, count=10),
            validation=SeedRangeConfig(start=9, count=10),
            test=SeedRangeConfig(start=20, count=10),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 2,
            "name": "bad",
            "seeds": [1],
            "seed_start": 2,
            "num_episodes": 1,
        },
        {"schema_version": 2, "name": "bad", "seed_start": 2},
        {"schema_version": 2, "name": "bad", "seeds": [1, 1]},
    ],
)
def test_evaluation_seed_sources_are_strict(tmp_path: Path, payload: dict[str, object]) -> None:
    payload["seed_partition"] = "test"
    payload["seed_partitions"] = {
        "train": {"start": 0, "count": 10},
        "validation": {"start": 10, "count": 10},
        "test": {"start": 20, "count": 10},
    }
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_evaluation_config(path)


def test_semantically_identical_generalization_configs_are_rejected(
    tmp_path: Path,
    easy_config: EnvConfig,
) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(easy_config.model_dump(mode="json")), encoding="utf-8")
    second.write_text(yaml.safe_dump(easy_config.model_dump(mode="json")), encoding="utf-8")
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "name": "identical",
                "seeds": [20],
                "seed_partition": "test",
                "seed_partitions": {
                    "train": {"start": 0, "count": 10},
                    "validation": {"start": 10, "count": 10},
                    "test": {"start": 20, "count": 10},
                },
                "comparison": {
                    "reference_environment": "first.yaml",
                    "held_out_environment": "second.yaml",
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation = load_evaluation_config(evaluation_path)
    with pytest.raises(ConfigurationError, match="semantically identical"):
        load_comparison_environment_configs(evaluation)
