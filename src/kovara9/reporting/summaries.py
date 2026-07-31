"""Generalization comparison summaries."""

from __future__ import annotations

from typing import Any

from kovara9.config.loader import configuration_fingerprint
from kovara9.config.models import EnvConfig
from kovara9.core.errors import ConfigurationError
from kovara9.evaluation.metrics import generalization_gap
from kovara9.evaluation.runner import EvaluationResult


def comparison_summary(
    reference: EvaluationResult,
    held_out: EvaluationResult,
    reference_config: EnvConfig,
    held_out_config: EnvConfig,
) -> dict[str, Any]:
    """Return an explicit structural generalization comparison."""

    reference_fingerprint = configuration_fingerprint(reference_config)
    held_out_fingerprint = configuration_fingerprint(held_out_config)
    if reference_fingerprint == held_out_fingerprint:
        raise ConfigurationError(
            "cannot report generalization for semantically identical environments"
        )
    return {
        "comparison_type": "structural_generalization",
        "metric": "success_rate",
        "definition": "reference_success_rate - held_out_success_rate",
        "reference_environment_fingerprint": reference_fingerprint,
        "held_out_environment_fingerprint": held_out_fingerprint,
        "reference_success_rate": reference.summary.metrics["success_rate"].mean,
        "held_out_success_rate": held_out.summary.metrics["success_rate"].mean,
        "generalization_gap": generalization_gap(reference.records, held_out.records),
    }
