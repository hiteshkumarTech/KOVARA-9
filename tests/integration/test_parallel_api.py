import pytest
from pettingzoo.test import parallel_api_test

from kovara9.config.models import CommunicationConfig, EnvConfig
from kovara9.environments.grid_rescue.environment import GridRescueParallelEnv


@pytest.mark.integration
def test_pettingzoo_parallel_api(easy_config: EnvConfig) -> None:
    config = easy_config.model_copy(
        update={
            "communication": CommunicationConfig(
                enabled=True,
                vocabulary_size=4,
                budget_per_agent=1000,
            )
        }
    )
    parallel_api_test(GridRescueParallelEnv(config), num_cycles=100)
