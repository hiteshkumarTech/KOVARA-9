from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kovara9.config.loader import load_environment_config, load_evaluation_config
from kovara9.config.models import CommunicationConfig, EnvConfig
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
    "payload",
    [
        {
            "schema_version": 1,
            "name": "bad",
            "seeds": [1],
            "seed_start": 2,
            "num_episodes": 1,
        },
        {"schema_version": 1, "name": "bad", "seed_start": 2},
        {"schema_version": 1, "name": "bad", "seeds": [1, 1]},
    ],
)
def test_evaluation_seed_sources_are_strict(tmp_path: Path, payload: dict[str, object]) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_evaluation_config(path)
