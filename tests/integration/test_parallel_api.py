from pathlib import Path

import pytest
from pettingzoo.test import parallel_api_test

from kovara9.config.loader import load_environment_config
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv


@pytest.mark.integration
@pytest.mark.parametrize("preset", ["easy", "medium", "hard"])
def test_pettingzoo_parallel_api_for_committed_presets(preset: str) -> None:
    config = load_environment_config(Path(f"configs/environments/grid_rescue_{preset}.yaml"))
    parallel_api_test(GridRescueParallelEnv(config), num_cycles=100)
