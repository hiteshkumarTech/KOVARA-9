"""One-time, preregistered Day 8 final held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch
from torch import Tensor

from kovara9.agents.frontier import FrontierPolicy
from kovara9.agents.random import RandomPolicy
from kovara9.config.loader import (
    configuration_fingerprint,
    load_comparison_environment_configs,
    load_evaluation_config,
    load_training_inputs,
)
from kovara9.config.models import EvaluationConfig
from kovara9.core.errors import ArtifactError, ConfigurationError, TrainingError
from kovara9.evaluation.runner import EvaluationResult, PolicyFactory, evaluate_policy
from kovara9.experiments.day6 import (
    DAY6_ROOT_SEEDS,
    load_candidate_freeze,
    validate_candidate_freeze,
)
from kovara9.reporting.artifacts import ArtifactWriter
from kovara9.reporting.summaries import comparison_summary
from kovara9.training.checkpoint import (
    LoadedCheckpoint,
    checkpoint_sha256,
    load_training_checkpoint,
    model_state_sha256,
    training_definition_fingerprint,
)
from kovara9.training.evaluation import actor_policy_factory
from kovara9.training.runtime import configure_deterministic_algorithms, resolve_device
from kovara9.training.trainer import (
    actor_from_checkpoint,
    untrained_actor_from_checkpoint_definition,
)

FINAL_POLICY_ORDER = (
    "random",
    "frontier",
    "untrained_seed_0",
    "untrained_seed_1",
    "untrained_seed_2",
    "trained_seed_0",
    "trained_seed_1",
    "trained_seed_2",
)
FinalClassification = Literal[
    "reproducible_task_learning_and_transfer",
    "weak_or_inconsistent_transfer",
    "exploration_transfer_without_task_completion",
    "no_meaningful_transfer",
    "negative_transfer",
    "evaluation_failure",
]


def file_sha256(path: Path) -> str:
    """Hash a complete file with contextual errors."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _update_state_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        descriptor = f"tensor:{tensor.dtype}:{tuple(tensor.shape)}".encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_state_digest(digest, key)
            _update_state_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"sequence:{type(value).__name__}:{len(value)}".encode())
        for item in value:
            _update_state_digest(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        digest.update(f"scalar:{type(value).__name__}:".encode())
        digest.update(encoded)
    else:
        raise TrainingError(f"cannot fingerprint optimizer value of type {type(value).__name__}")


def optimizer_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash a nested optimizer state without relying on container serialization."""

    digest = hashlib.sha256()
    _update_state_digest(digest, state)
    return digest.hexdigest()


def checkpoint_integrity(path: Path, loaded: LoadedCheckpoint | None = None) -> dict[str, str]:
    """Return all immutable checkpoint identities used by final evaluation."""

    checkpoint = loaded or load_training_checkpoint(path)
    return {
        "checkpoint_sha256": checkpoint_sha256(path),
        "actor_state_sha256": model_state_sha256(checkpoint.actor_state),
        "critic_state_sha256": model_state_sha256(checkpoint.critic_state),
        "optimizer_state_sha256": optimizer_state_sha256(checkpoint.optimizer_state),
    }


def verify_preregistration(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify the immutable protocol bytes and return its JSON object."""

    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ConfigurationError(
            f"preregistration fingerprint mismatch: expected={expected_sha256}, actual={actual}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load Day 8 preregistration {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("status") != "preregistered":
        raise ConfigurationError("Day 8 preregistration is missing or not preregistered")
    return raw


def reject_consumed_test_partition(
    evaluation: EvaluationConfig,
    consumed_record: Path = Path("configs/evaluation/final_test_consumed.json"),
) -> None:
    """Prevent generic evaluation/tuning commands from reusing consumed final seeds."""

    if evaluation.seed_partition == "test" and consumed_record.exists():
        raise ConfigurationError(
            "final test partition is consumed; tuning and ordinary evaluation cannot reuse it"
        )


def validate_policy_seed_alignment(
    results: Mapping[str, Mapping[str, EvaluationResult]],
    expected_seeds: tuple[int, ...],
) -> None:
    """Require every declared policy/environment suite and exact ordered seed alignment."""

    if tuple(results) != FINAL_POLICY_ORDER:
        raise TrainingError(
            f"final policy order mismatch: expected={FINAL_POLICY_ORDER}, actual={tuple(results)}"
        )
    for policy, environments in results.items():
        if tuple(environments) != ("reference", "structural"):
            raise TrainingError(f"final environments are incomplete for policy {policy}")
        for environment, result in environments.items():
            actual = tuple(record.seed for record in result.records)
            if actual != expected_seeds:
                raise TrainingError(
                    f"final seed alignment mismatch for {policy}/{environment}: {actual}"
                )


def generalization_gap(validation_metric: float, held_out_metric: float) -> float:
    """Return validation minus held-out; positive values mean degradation."""

    if not math.isfinite(validation_metric) or not math.isfinite(held_out_metric):
        raise ValueError("generalization-gap inputs must be finite")
    return validation_metric - held_out_metric


def classify_final_performance(  # noqa: PLR0911
    trained: Mapping[int, Mapping[str, float]],
    untrained: Mapping[int, Mapping[str, float]],
    random: Mapping[str, float],
    *,
    valid: bool = True,
) -> FinalClassification:
    """Apply the preregistered ordered final classification rules."""

    if not valid or set(trained) != set(DAY6_ROOT_SEEDS) or set(untrained) != set(DAY6_ROOT_SEEDS):
        return "evaluation_failure"
    required = ("success_rate", "targets_recovered", "completion_progress", "exploration_coverage")
    values = [
        metrics[name]
        for collection in (trained, untrained)
        for metrics in collection.values()
        for name in required
    ] + [random[name] for name in required[:3]]
    if not all(math.isfinite(value) for value in values):
        return "evaluation_failure"

    differences = {
        name: tuple(trained[seed][name] - untrained[seed][name] for seed in DAY6_ROOT_SEEDS)
        for name in required
    }
    trained_mean = {
        name: sum(trained[seed][name] for seed in DAY6_ROOT_SEEDS) / len(DAY6_ROOT_SEEDS)
        for name in required
    }
    reproducible = (
        all(value > 0.0 for value in differences["success_rate"])
        and trained_mean["success_rate"] > random["success_rate"]
        and trained_mean["targets_recovered"] > random["targets_recovered"]
        and trained_mean["completion_progress"] > random["completion_progress"]
        and all(value > 0.0 for value in differences["targets_recovered"])
        and all(value > 0.0 for value in differences["completion_progress"])
    )
    if reproducible:
        return "reproducible_task_learning_and_transfer"

    success_mean_difference = sum(differences["success_rate"]) / len(DAY6_ROOT_SEEDS)
    completion_differences = differences["completion_progress"]
    negative = success_mean_difference < 0.0 or (
        success_mean_difference == 0.0
        and sum(completion_differences) < 0.0
        and sum(value <= 0.0 for value in completion_differences) >= 2
        and any(value < 0.0 for value in completion_differences)
    )
    if negative:
        return "negative_transfer"
    partial_improvements = tuple(
        differences["exploration_coverage"][index] > 0.0
        or differences["targets_recovered"][index] > 0.0
        for index in range(len(DAY6_ROOT_SEEDS))
    )
    if (
        all(trained[seed]["success_rate"] == 0.0 for seed in DAY6_ROOT_SEEDS)
        and sum(partial_improvements) >= 2
    ):
        return "exploration_transfer_without_task_completion"
    if any(
        differences[name][index] > 0.0
        for name in ("success_rate", "targets_recovered", "completion_progress")
        for index in range(len(DAY6_ROOT_SEEDS))
    ):
        return "weak_or_inconsistent_transfer"
    return "no_meaningful_transfer"


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if exclusive:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        else:
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.replace(path)
    except FileExistsError as exc:
        raise ArtifactError(f"final test partition is already claimed: {path}") from exc
    except OSError as exc:
        raise ArtifactError(f"cannot write final evaluation record {path}: {exc}") from exc


def claim_final_test(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically claim the final partition exactly once."""

    _atomic_json(path, payload, exclusive=True)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True, timeout=5
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _policy_factory_for_checkpoint(
    policy_id: str,
    loaded: LoadedCheckpoint,
    environment: Any,
    device: torch.device,
    identity: Mapping[str, str],
) -> PolicyFactory:
    actor = actor_from_checkpoint(loaded, environment=environment, device=device)
    return actor_policy_factory(
        actor=actor,
        device=device,
        policy_name=policy_id,
        parameters={
            **identity,
            "deterministic": True,
            "training_seed": loaded.metadata.training_config.seed,
            "environment_steps": loaded.metadata.progress.environment_steps,
        },
    )


def _evaluate_pair(
    *,
    policy_factory: PolicyFactory,
    evaluation: EvaluationConfig,
    reference: Any,
    structural: Any,
    output: Path,
) -> dict[str, EvaluationResult]:
    reference_result = evaluate_policy(
        env_config=reference, evaluation_config=evaluation, policy_factory=policy_factory
    )
    structural_result = evaluate_policy(
        env_config=structural, evaluation_config=evaluation, policy_factory=policy_factory
    )
    ArtifactWriter(output).write(
        env_config=reference,
        evaluation_config=evaluation,
        result=reference_result,
        held_out_env_config=structural,
        held_out_result=structural_result,
        comparison=comparison_summary(reference_result, structural_result, reference, structural),
    )
    return {"reference": reference_result, "structural": structural_result}


def run_final_evaluation(  # noqa: PLR0912, PLR0913, PLR0915
    *,
    candidate_path: Path,
    freeze_path: Path,
    evaluation_path: Path,
    preregistration_path: Path,
    preregistration_sha256: str,
    artifact_root: Path,
    output: Path,
    consumed_record: Path,
    device_name: str,
) -> dict[str, Any]:
    """Preflight, claim, and execute the sole preregistered held-out evaluation."""

    preregistration = verify_preregistration(preregistration_path, preregistration_sha256)
    candidate_inputs = load_training_inputs(candidate_path)
    validate_candidate_freeze(candidate_path, load_candidate_freeze(freeze_path))
    evaluation = load_evaluation_config(evaluation_path)
    reject_consumed_test_partition(evaluation, consumed_record)
    reference, structural = load_comparison_environment_configs(evaluation)
    if output.exists():
        raise ArtifactError(f"final evaluation output already exists: {output}")

    declared_seeds = tuple(preregistration["seed_partitions"]["final_test_seeds"])
    if evaluation.resolved_seeds != declared_seeds:
        raise ConfigurationError("evaluation seeds do not match the preregistration")
    if (
        configuration_fingerprint(evaluation)
        != preregistration["evaluation"]["configuration_fingerprint"]
    ):
        raise ConfigurationError("evaluation configuration changed after preregistration")
    if (
        configuration_fingerprint(reference)
        != preregistration["evaluation"]["reference_environment"]["fingerprint"]
        or configuration_fingerprint(structural)
        != preregistration["evaluation"]["structural_environment"]["fingerprint"]
    ):
        raise ConfigurationError("held-out environment changed after preregistration")

    policy_specs = {spec["id"]: spec for spec in preregistration["policies"]}
    if tuple(policy_specs) != FINAL_POLICY_ORDER:
        raise ConfigurationError("preregistered final policy order is incomplete or changed")
    loaded_by_policy: dict[str, LoadedCheckpoint] = {}
    checkpoint_paths: dict[str, Path] = {}
    before: dict[str, dict[str, str]] = {}
    for policy_id in FINAL_POLICY_ORDER[2:]:
        spec = policy_specs[policy_id]
        path = (artifact_root / spec["checkpoint"]).resolve()
        if not path.is_relative_to(artifact_root.resolve()):
            raise ConfigurationError(f"checkpoint escapes artifact root: {path}")
        loaded = load_training_checkpoint(path)
        identity = checkpoint_integrity(path, loaded)
        for name in ("checkpoint_sha256", "actor_state_sha256", "critic_state_sha256"):
            if identity[name] != spec[name]:
                raise ConfigurationError(f"{policy_id} {name} changed after preregistration")
        seed = int(spec["training_seed"])
        expected_training = candidate_inputs.training.model_copy(update={"seed": seed})
        if (
            loaded.metadata.training_config.seed != seed
            or loaded.metadata.training_fingerprint
            != training_definition_fingerprint(expected_training)
            or loaded.metadata.environment_fingerprint
            != preregistration["candidate"]["environment_fingerprint"]
            or loaded.metadata.validation_fingerprint
            != preregistration["candidate"]["validation_fingerprint"]
        ):
            raise ConfigurationError(f"{policy_id} checkpoint provenance does not match candidate")
        expected_steps = (
            0 if policy_id.startswith("untrained") else spec["selected_environment_steps"]
        )
        if loaded.metadata.progress.environment_steps != expected_steps:
            raise ConfigurationError(f"{policy_id} selected checkpoint step changed")
        loaded_by_policy[policy_id] = loaded
        checkpoint_paths[policy_id] = path
        before[policy_id] = identity

    for seed in DAY6_ROOT_SEEDS:
        untrained = loaded_by_policy[f"untrained_seed_{seed}"]
        trained = loaded_by_policy[f"trained_seed_{seed}"]
        recreated = untrained_actor_from_checkpoint_definition(
            trained, environment=reference, device=torch.device("cpu")
        )
        if model_state_sha256(recreated.state_dict()) != model_state_sha256(untrained.actor_state):
            raise ConfigurationError(f"seed {seed} saved initialization is not exact")

    git = _git_state()
    if git["dirty"]:
        raise ConfigurationError("scientific snapshot must be clean before final-test consumption")
    started_at = datetime.now(UTC).isoformat()
    claim: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "started_at_utc": started_at,
        "git": git,
        "candidate_fingerprint": preregistration["candidate"]["configuration_fingerprint"],
        "preregistration_sha256": preregistration_sha256,
        "evaluation_fingerprint": configuration_fingerprint(evaluation),
        "test_seeds": list(declared_seeds),
        "environments": {
            "reference": configuration_fingerprint(reference),
            "structural": configuration_fingerprint(structural),
        },
        "exact_command": [sys.executable, *sys.argv],
        "checkpoint_integrity_before": before,
        "completed_policies": [],
    }
    claim_final_test(consumed_record, claim)
    started = perf_counter()
    results: dict[str, dict[str, EvaluationResult]] = {}
    try:
        output.mkdir(parents=True, exist_ok=False)
        device = resolve_device(device_name)  # type: ignore[arg-type]
        configure_deterministic_algorithms(True)
        factories: dict[str, PolicyFactory] = {"random": RandomPolicy, "frontier": FrontierPolicy}
        for policy_id in FINAL_POLICY_ORDER[2:]:
            factories[policy_id] = _policy_factory_for_checkpoint(
                policy_id,
                loaded_by_policy[policy_id],
                reference,
                device,
                before[policy_id],
            )
        for policy_id in FINAL_POLICY_ORDER:
            rng_before = torch.random.get_rng_state().clone()
            results[policy_id] = _evaluate_pair(
                policy_factory=factories[policy_id],
                evaluation=evaluation,
                reference=reference,
                structural=structural,
                output=output / policy_id,
            )
            if policy_id not in ("random", "frontier") and not torch.equal(
                rng_before, torch.random.get_rng_state()
            ):
                raise TrainingError(f"neural evaluation changed Torch RNG state for {policy_id}")
            claim["completed_policies"] = [*claim["completed_policies"], policy_id]
            _atomic_json(consumed_record, claim)
        validate_policy_seed_alignment(results, declared_seeds)
        verify_preregistration(preregistration_path, preregistration_sha256)
        after = {
            policy_id: checkpoint_integrity(checkpoint_paths[policy_id])
            for policy_id in FINAL_POLICY_ORDER[2:]
        }
        if after != before:
            raise TrainingError(
                "checkpoint, actor, critic, or optimizer state changed during evaluation"
            )
        index = {
            "schema_version": 1,
            "status": "complete",
            "preregistration_sha256": preregistration_sha256,
            "policy_order": list(FINAL_POLICY_ORDER),
            "environment_order": ["reference", "structural"],
            "test_seeds": list(declared_seeds),
            "checkpoint_integrity_before": before,
            "checkpoint_integrity_after": after,
            "runtime_seconds": perf_counter() - started,
        }
        _atomic_json(output / "evaluation-index.json", index)
        artifact_checksums = {
            str(path.relative_to(output)).replace("\\", "/"): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        claim.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "runtime_seconds": index["runtime_seconds"],
                "checkpoint_integrity_after": after,
                "artifact_checksums": artifact_checksums,
            }
        )
        _atomic_json(consumed_record, claim)
        return claim
    except Exception as exc:
        claim.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "failure": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": perf_counter() - started,
            }
        )
        _atomic_json(consumed_record, claim)
        raise
