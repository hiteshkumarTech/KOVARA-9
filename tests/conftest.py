from pathlib import Path

import pytest

from kovara9.config.loader import load_environment_config
from kovara9.config.models import EnvConfig


@pytest.fixture
def easy_config() -> EnvConfig:
    return load_environment_config(Path("configs/environments/grid_rescue_easy.yaml"))
