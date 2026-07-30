"""Generalization comparison summaries."""

from __future__ import annotations

from typing import Any

from kovara9.evaluation.metrics import generalization_gap
from kovara9.evaluation.runner import EvaluationResult


def comparison_summary(
    reference: EvaluationResult,
    held_out: EvaluationResult,
) -> dict[str, Any]:
    """Return an explicit structural generalization comparison."""

    return {
        "metric": "success_rate",
        "definition": "reference_success_rate - held_out_success_rate",
        "reference_success_rate": reference.summary.metrics["success_rate"].mean,
        "held_out_success_rate": held_out.summary.metrics["success_rate"].mean,
        "generalization_gap": generalization_gap(reference.records, held_out.records),
    }
