"""Dependency-light RGB renderer."""

import numpy as np
from numpy.typing import NDArray

from kovara9.core.types import WorldSnapshot


class RgbArrayRenderer:
    """Render one colored pixel per world cell."""

    FLOOR = np.array([235, 238, 240], dtype=np.uint8)
    OBSTACLE = np.array([45, 52, 54], dtype=np.uint8)
    TARGET = np.array([220, 70, 70], dtype=np.uint8)
    RECOVERED = np.array([80, 160, 95], dtype=np.uint8)
    AGENT_COLORS = (
        np.array([50, 110, 220], dtype=np.uint8),
        np.array([230, 150, 35], dtype=np.uint8),
        np.array([145, 75, 200], dtype=np.uint8),
        np.array([30, 175, 175], dtype=np.uint8),
    )

    def render(self, snapshot: WorldSnapshot) -> NDArray[np.uint8]:
        """Return an H-by-W-by-3 defensive RGB array."""

        image = np.empty((snapshot.height, snapshot.width, 3), dtype=np.uint8)
        image[:] = self.FLOOR
        image[snapshot.obstacles] = self.OBSTACLE
        for target in snapshot.targets:
            color = self.RECOVERED if target in snapshot.recovered_targets else self.TARGET
            image[target.row, target.col] = color
        for slot, agent in enumerate(sorted(snapshot.agent_positions)):
            position = snapshot.agent_positions[agent]
            image[position.row, position.col] = self.AGENT_COLORS[slot]
        return image
