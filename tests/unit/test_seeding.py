import numpy as np
import pytest

from kovara9.core.seeding import derive_seed, make_rng


def test_rng_is_reproducible_and_local() -> None:
    np.random.seed(123)
    before = np.random.get_state()
    first = make_rng(7).integers(0, 1000, size=20)
    second = make_rng(7).integers(0, 1000, size=20)
    after = np.random.get_state()
    assert np.array_equal(first, second)
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])


def test_semantic_child_seeds_are_stable_and_distinct() -> None:
    assert derive_seed(4, "agent", 1) == derive_seed(4, "agent", 1)
    assert derive_seed(4, "agent", 1) != derive_seed(4, "agent", 2)


@pytest.mark.parametrize("function", [make_rng, derive_seed])
def test_negative_seeds_are_rejected(function: object) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        function(-1)  # type: ignore[operator]
