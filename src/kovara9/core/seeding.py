"""Deterministic seed derivation without global random state."""

from __future__ import annotations

import hashlib

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a local generator for one explicitly seeded operation."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    return np.random.default_rng(np.random.SeedSequence(seed))


def derive_seed(parent_seed: int, *labels: str | int) -> int:
    """Derive a stable unsigned 64-bit child seed from semantic labels."""

    if parent_seed < 0:
        raise ValueError("parent_seed must be non-negative")
    payload = "\x1f".join([str(parent_seed), *(str(label) for label in labels)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
