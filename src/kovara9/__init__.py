"""KOVARA-9 multi-agent embodied-AI research platform."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kovara9")
except PackageNotFoundError:  # pragma: no cover - source trees without installation
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
