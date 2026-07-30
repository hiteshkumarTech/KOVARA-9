import numpy as np

from kovara9.core.types import Position, WorldSnapshot
from kovara9.rendering.ansi import AnsiRenderer
from kovara9.rendering.rgb_array import RgbArrayRenderer


def _snapshot() -> WorldSnapshot:
    obstacles = np.zeros((3, 4), dtype=np.bool_)
    obstacles[0, 0] = True
    obstacles.flags.writeable = False
    return WorldSnapshot(
        width=4,
        height=3,
        obstacles=obstacles,
        agent_positions={"agent_0": Position(1, 1)},
        targets=frozenset({Position(2, 2), Position(0, 3)}),
        recovered_targets=frozenset({Position(2, 2)}),
        communication_budgets={"agent_0": 2},
        latest_messages={"agent_0": 0},
        step_count=5,
    )


def test_ansi_renderer_is_deterministic() -> None:
    rendered = AnsiRenderer().render(_snapshot())
    assert rendered == "#..T\n.0..\n..x.\nstep=5 recovered=1/2"


def test_rgb_renderer_shape_colors_and_defensive_output() -> None:
    snapshot = _snapshot()
    before = snapshot.obstacles.tobytes()
    image = RgbArrayRenderer().render(snapshot)
    assert image.shape == (3, 4, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image[0, 0], RgbArrayRenderer.OBSTACLE)
    assert np.array_equal(image[1, 1], RgbArrayRenderer.AGENT_COLORS[0])
    image[:] = 0
    assert snapshot.obstacles.tobytes() == before
