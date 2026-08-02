"""Semantic deterministic seed streams for one training experiment."""

from __future__ import annotations

from dataclasses import dataclass

from kovara9.core.seeding import derive_seed


@dataclass(frozen=True, slots=True)
class ExperimentSeedStreams:
    """Derive independent child streams without persistent use of ``hash()``."""

    root_seed: int

    def __post_init__(self) -> None:
        if self.root_seed < 0:
            raise ValueError("root experiment seed must be non-negative")

    @property
    def actor_initialization(self) -> int:
        """Seed the shared actor's parameter initialization."""

        return derive_seed(self.root_seed, "network", "actor")

    @property
    def critic_initialization(self) -> int:
        """Seed the centralized critic's parameter initialization."""

        return derive_seed(self.root_seed, "network", "critic")

    @property
    def policy_sampling(self) -> int:
        """Seed the on-policy action-sampling stream."""

        return derive_seed(self.root_seed, "policy", "sampling")

    @property
    def optimizer_shuffle(self) -> int:
        """Seed deterministic PPO epoch and minibatch shuffling."""

        return derive_seed(self.root_seed, "optimizer", "shuffle")

    def environment_instance(self, environment_id: int) -> int:
        """Identify one environment instance independently of reset order."""

        self._validate_index("environment_id", environment_id)
        return derive_seed(self.root_seed, "environment", environment_id)

    def environment_reset(self, environment_id: int, episode_index: int) -> int:
        """Seed a specific episode in one environment's independent stream."""

        self._validate_index("environment_id", environment_id)
        self._validate_index("episode_index", episode_index)
        return derive_seed(
            self.environment_instance(environment_id),
            "episode",
            episode_index,
        )

    def evaluation(self, evaluation_index: int) -> int:
        """Seed a deterministic evaluation operation."""

        self._validate_index("evaluation_index", evaluation_index)
        return derive_seed(self.root_seed, "evaluation", evaluation_index)

    @staticmethod
    def _validate_index(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
