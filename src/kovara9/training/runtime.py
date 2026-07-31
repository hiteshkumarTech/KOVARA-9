"""Explicit device and PyTorch reproducibility controls."""

from __future__ import annotations

import torch

from kovara9.core.errors import ConfigurationError
from kovara9.training.config import DeviceName


def resolve_device(requested: DeviceName) -> torch.device:
    """Resolve an explicit or automatic device without silently falling back."""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("training device is cuda but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def make_torch_generator(seed: int, device: torch.device) -> torch.Generator:
    """Create one explicitly seeded generator for a declared stochastic stream."""

    if seed < 0:
        raise ValueError("torch generator seed must be non-negative")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def configure_deterministic_algorithms(enabled: bool) -> None:
    """Configure deterministic kernels for this recorded training process."""

    torch.use_deterministic_algorithms(enabled)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not enabled
        torch.backends.cudnn.deterministic = enabled
