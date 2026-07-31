"""Renderer extension boundary."""

from __future__ import annotations

from typing import Protocol, TypeVar

from kovara9.core.types import WorldSnapshot

RenderOutputT_co = TypeVar("RenderOutputT_co", covariant=True)


class Renderer(Protocol[RenderOutputT_co]):
    """Pure renderer consuming an immutable snapshot."""

    def render(self, snapshot: WorldSnapshot) -> RenderOutputT_co:
        """Return a representation without mutating the simulator."""

        ...
