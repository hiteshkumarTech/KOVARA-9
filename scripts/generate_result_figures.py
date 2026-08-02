"""Generate deterministic SVG figures from the committed Day 8 result report only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPOSITORY_ROOT / "docs/day8-final-heldout-results.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "docs/assets/results"

POLICY_ORDER = (
    "random",
    "frontier",
    "untrained_seed_0",
    "untrained_seed_1",
    "untrained_seed_2",
    "trained_seed_0",
    "trained_seed_1",
    "trained_seed_2",
)
POLICY_LABELS = {
    "random": "Random",
    "frontier": "Frontier",
    "untrained_seed_0": "Init 0",
    "untrained_seed_1": "Init 1",
    "untrained_seed_2": "Init 2",
    "trained_seed_0": "Trained 0",
    "trained_seed_1": "Trained 1",
    "trained_seed_2": "Trained 2",
}
POLICY_COLORS = {
    "random": "#64748b",
    "frontier": "#0f766e",
    "untrained_seed_0": "#cbd5e1",
    "untrained_seed_1": "#cbd5e1",
    "untrained_seed_2": "#cbd5e1",
    "trained_seed_0": "#7c3aed",
    "trained_seed_1": "#7c3aed",
    "trained_seed_2": "#7c3aed",
}
TEXT_COLOR = "#172033"
GRID_COLOR = "#dbe3ef"
BACKGROUND = "#ffffff"
ACCENT = "#7c3aed"


@dataclass(frozen=True)
class Bar:
    """One labeled value in an SVG chart."""

    label: str
    value: float
    color: str


@dataclass(frozen=True)
class Panel:
    """One independently scaled panel in a small-multiple figure."""

    title: str
    unit: str
    bars: tuple[Bar, ...]


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _float(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _nested(mapping: Mapping[str, Any], *keys: str) -> object:
    current: object = mapping
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        current = _mapping(current, ".".join(traversed[:-1]) or "report").get(key)
        if current is None:
            raise ValueError(f"missing result value: {'.'.join(traversed)}")
    return current


def load_report(path: Path) -> dict[str, Any]:
    """Load and minimally validate the authoritative final-results JSON."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load final-results JSON {path}: {exc}") from exc
    report = dict(_mapping(raw, "report"))
    if report.get("status") != "complete":
        raise ValueError("final-results JSON is not complete")
    if report.get("classification") != "exploration_transfer_without_task_completion":
        raise ValueError("unexpected final-results classification")
    return report


def source_sha256(path: Path) -> str:
    """Return the byte-level SHA-256 for the figure source."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot hash figure source {path}: {exc}") from exc
    return digest.hexdigest()


def build_figure_data(report: Mapping[str, Any]) -> dict[str, Any]:
    """Select figure values directly from the final report without recomputation."""

    policies = _mapping(_nested(report, "policy_results"), "policy_results")
    missing = [name for name in POLICY_ORDER if name not in policies]
    if missing:
        raise ValueError(f"final report is missing required policies: {missing}")

    policy_metrics: dict[str, dict[str, float]] = {}
    for name in POLICY_ORDER:
        pooled = _mapping(_nested(policies, name, "pooled"), f"policy_results.{name}.pooled")
        policy_metrics[name] = {
            metric: _float(pooled.get(metric), f"policy_results.{name}.pooled.{metric}")
            for metric in ("success_rate", "targets_recovered", "exploration_coverage")
        }

    paired_source = _mapping(
        _nested(report, "paired_trained_minus_untrained"),
        "paired_trained_minus_untrained",
    )
    paired = {
        f"seed_{seed}": {
            metric: _float(
                _nested(paired_source, f"seed_{seed}", metric),
                f"paired_trained_minus_untrained.seed_{seed}.{metric}",
            )
            for metric in ("targets_recovered", "exploration_coverage")
        }
        for seed in range(3)
    }

    trained = _mapping(
        _nested(report, "training_seed_aggregates", "trained"),
        "training_seed_aggregates.trained",
    )
    baseline = {
        "random": {
            metric: _float(
                _nested(policies, "random", "pooled", metric),
                f"policy_results.random.pooled.{metric}",
            )
            for metric in ("success_rate", "completion_progress", "exploration_coverage")
        },
        "frontier": {
            metric: _float(
                _nested(policies, "frontier", "pooled", metric),
                f"policy_results.frontier.pooled.{metric}",
            )
            for metric in ("success_rate", "completion_progress", "exploration_coverage")
        },
        "trained_mean": {
            metric: _float(
                _nested(trained, metric, "mean"),
                f"training_seed_aggregates.trained.{metric}.mean",
            )
            for metric in ("success_rate", "completion_progress", "exploration_coverage")
        },
    }

    gaps_source = _mapping(_nested(report, "generalization_gap"), "generalization_gap")
    gaps = {
        f"seed_{seed}": {
            metric: _float(
                _nested(gaps_source, f"seed_{seed}", metric),
                f"generalization_gap.seed_{seed}.{metric}",
            )
            for metric in (
                "success_rate",
                "targets_recovered",
                "completion_progress",
                "exploration_coverage",
            )
        }
        for seed in range(3)
    }
    return {
        "policy_order": list(POLICY_ORDER),
        "policy_metrics": policy_metrics,
        "paired_trained_minus_untrained": paired,
        "baseline_comparison": baseline,
        "generalization_gaps": gaps,
    }


def _format_number(value: float, unit: str) -> str:
    if unit == "rate":
        return f"{value:.1%}"
    if unit == "delta":
        return f"{value:+.3f}"
    return f"{value:.3f}"


def _svg_text(  # noqa: PLR0913
    x: float,
    y: float,
    content: str,
    *,
    anchor: str = "start",
    size: int = 13,
    weight: int = 400,
    fill: str = TEXT_COLOR,
) -> str:
    return (
        f'  <text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}">{escape(content)}</text>'
    )


def _svg_line(  # noqa: PLR0913
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float,
) -> str:
    return (
        f'  <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"/>'
    )


def _svg_rect(  # noqa: PLR0913
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    radius: int,
    data_policy: str | None = None,
) -> str:
    policy_attribute = "" if data_policy is None else f' data-policy="{escape(data_policy)}"'
    return (
        f'  <rect{policy_attribute} x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="{radius}" fill="{fill}"/>'
    )


def _svg_header(title: str, description: str, width: int, height: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f'  <title id="title">{escape(title)}</title>',
        f'  <desc id="desc">{escape(description)}</desc>',
        f'  <rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        _svg_text(60, 52, title, size=28, weight=700),
    ]


def _provenance(lines: list[str], source_hash: str, height: int) -> None:
    note = (
        "Source: docs/day8-final-heldout-results.json · deterministic SVG · "
        f"SHA-256 {source_hash[:16]}…"
    )
    lines.append(_svg_text(60, height - 24, note, fill="#64748b"))
    lines.append("</svg>")


def _axis_bounds(
    values: Sequence[float], *, fixed_maximum: float | None = None
) -> tuple[float, float]:
    if fixed_maximum is not None:
        return 0.0, fixed_maximum
    minimum = min(0.0, *values)
    maximum = max(0.0, *values)
    span = maximum - minimum
    padding = max(span * 0.12, 0.02)
    if span == 0.0:
        return -0.05, 0.05
    return minimum - (padding if minimum < 0.0 else 0.0), maximum + padding


def _bar_chart(  # noqa: PLR0913
    *,
    title: str,
    description: str,
    bars: Sequence[Bar],
    unit: str,
    source_hash: str,
    fixed_maximum: float | None = None,
) -> str:
    width, height = 1200, 700
    left, top, plot_width, plot_height = 80, 100, 1080, 440
    minimum, maximum = _axis_bounds([bar.value for bar in bars], fixed_maximum=fixed_maximum)
    lines = _svg_header(title, description, width, height)
    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        y = top + plot_height - (value - minimum) / (maximum - minimum) * plot_height
        lines.extend(
            [
                _svg_line(left, y, left + plot_width, y, stroke=GRID_COLOR, width=1),
                _svg_text(
                    left - 10,
                    y + 5,
                    _format_number(value, unit),
                    anchor="end",
                    fill="#475569",
                ),
            ]
        )
    zero_y = top + plot_height - (0.0 - minimum) / (maximum - minimum) * plot_height
    lines.append(_svg_line(left, zero_y, left + plot_width, zero_y, stroke="#334155", width=1.5))
    slot = plot_width / len(bars)
    bar_width = slot * 0.62
    for index, bar in enumerate(bars):
        x = left + slot * index + (slot - bar_width) / 2
        value_y = top + plot_height - (bar.value - minimum) / (maximum - minimum) * plot_height
        y = min(value_y, zero_y)
        visual_height = max(abs(zero_y - value_y), 2.0)
        lines.extend(
            [
                _svg_rect(
                    x,
                    y,
                    bar_width,
                    visual_height,
                    fill=bar.color,
                    radius=4,
                    data_policy=POLICY_ORDER[index],
                ),
                _svg_text(
                    x + bar_width / 2,
                    min(y - 8, zero_y - 8),
                    _format_number(bar.value, unit),
                    anchor="middle",
                    weight=600,
                ),
                _svg_text(
                    x + bar_width / 2,
                    top + plot_height + 28,
                    bar.label,
                    anchor="middle",
                ),
            ]
        )
    _provenance(lines, source_hash, height)
    return "\n".join(lines) + "\n"


def _panel_chart(
    *,
    title: str,
    description: str,
    panels: Sequence[Panel],
    source_hash: str,
) -> str:
    width, height = 1200, 680
    lines = _svg_header(title, description, width, height)
    outer_left, panel_gap = 50, 28
    panel_width = (width - outer_left * 2 - panel_gap * (len(panels) - 1)) / len(panels)
    for panel_index, panel in enumerate(panels):
        panel_x = outer_left + panel_index * (panel_width + panel_gap)
        chart_left = panel_x + 55
        chart_top = 140
        chart_width = panel_width - 75
        chart_height = 350
        values = [bar.value for bar in panel.bars]
        minimum, maximum = _axis_bounds(
            values,
            fixed_maximum=1.0 if panel.unit == "rate" else None,
        )
        lines.append(
            _svg_text(
                panel_x + panel_width / 2,
                105,
                panel.title,
                anchor="middle",
                size=18,
                weight=700,
            )
        )
        for tick in range(5):
            value = minimum + (maximum - minimum) * tick / 4
            y = chart_top + chart_height - (value - minimum) / (maximum - minimum) * chart_height
            lines.extend(
                [
                    _svg_line(
                        chart_left,
                        y,
                        chart_left + chart_width,
                        y,
                        stroke=GRID_COLOR,
                        width=1,
                    ),
                    _svg_text(
                        chart_left - 8,
                        y + 4,
                        _format_number(value, panel.unit),
                        anchor="end",
                        size=11,
                        fill="#64748b",
                    ),
                ]
            )
        zero_y = chart_top + chart_height - (0.0 - minimum) / (maximum - minimum) * chart_height
        lines.append(
            _svg_line(
                chart_left,
                zero_y,
                chart_left + chart_width,
                zero_y,
                stroke="#334155",
                width=1.5,
            )
        )
        slot = chart_width / len(panel.bars)
        bar_width = slot * 0.58
        for bar_index, bar in enumerate(panel.bars):
            x = chart_left + slot * bar_index + (slot - bar_width) / 2
            value_y = (
                chart_top
                + chart_height
                - (bar.value - minimum) / (maximum - minimum) * chart_height
            )
            y = min(value_y, zero_y)
            visual_height = max(abs(zero_y - value_y), 2.0)
            label_y = y - 8 if bar.value >= 0 else y + visual_height + 17
            lines.extend(
                [
                    _svg_rect(
                        x,
                        y,
                        bar_width,
                        visual_height,
                        fill=bar.color,
                        radius=3,
                    ),
                    _svg_text(
                        x + bar_width / 2,
                        label_y,
                        _format_number(bar.value, panel.unit),
                        anchor="middle",
                        size=11,
                        weight=600,
                    ),
                    _svg_text(
                        x + bar_width / 2,
                        chart_top + chart_height + 26,
                        bar.label,
                        anchor="middle",
                        size=12,
                    ),
                ]
            )
    _provenance(lines, source_hash, height)
    return "\n".join(lines) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def generate_figures(report_path: Path, output_directory: Path) -> dict[str, Any]:
    """Generate all committed figures and return their data manifest."""

    report = load_report(report_path)
    data = build_figure_data(report)
    source_hash = source_sha256(report_path)
    policy_metrics = _mapping(data["policy_metrics"], "figure_data.policy_metrics")

    for metric, title, description, unit, maximum, filename in (
        (
            "success_rate",
            "Held-out success rate by policy",
            "All frozen learned and exact-untrained actors had zero full-task success.",
            "rate",
            1.0,
            "success-rate-by-policy.svg",
        ),
        (
            "targets_recovered",
            "Mean held-out targets recovered by policy",
            "Pooled mean over the reference and structural held-out suites.",
            "value",
            None,
            "targets-recovered-by-policy.svg",
        ),
        (
            "exploration_coverage",
            "Held-out exploration coverage by policy",
            "Coverage transferred partially but remained below random and frontier baselines.",
            "rate",
            1.0,
            "exploration-coverage-by-policy.svg",
        ),
    ):
        bars = tuple(
            Bar(
                label=POLICY_LABELS[name],
                value=_float(
                    _nested(policy_metrics, name, metric),
                    f"figure_data.policy_metrics.{name}.{metric}",
                ),
                color=POLICY_COLORS[name],
            )
            for name in POLICY_ORDER
        )
        _write_text(
            output_directory / filename,
            _bar_chart(
                title=title,
                description=description,
                bars=bars,
                unit=unit,
                source_hash=source_hash,
                fixed_maximum=maximum,
            ),
        )

    paired = _mapping(
        data["paired_trained_minus_untrained"],
        "figure_data.paired_trained_minus_untrained",
    )
    paired_panels = tuple(
        Panel(
            title=title,
            unit="delta",
            bars=tuple(
                Bar(
                    label=f"Seed {seed}",
                    value=_float(
                        _nested(paired, f"seed_{seed}", metric),
                        f"figure_data.paired.seed_{seed}.{metric}",
                    ),
                    color=ACCENT,
                )
                for seed in range(3)
            ),
        )
        for metric, title in (
            ("targets_recovered", "Targets recovered Δ"),
            ("exploration_coverage", "Coverage Δ"),
        )
    )
    _write_text(
        output_directory / "trained-minus-untrained.svg",
        _panel_chart(
            title="Trained minus exact-untrained, paired by seed",
            description=(
                "Positive values indicate partial-behavior improvement; "
                "success differences were zero."
            ),
            panels=paired_panels,
            source_hash=source_hash,
        ),
    )

    baseline = _mapping(data["baseline_comparison"], "figure_data.baseline_comparison")
    baseline_panels = tuple(
        Panel(
            title=title,
            unit="rate",
            bars=tuple(
                Bar(
                    label=label,
                    value=_float(
                        _nested(baseline, name, metric),
                        f"figure_data.baseline.{name}.{metric}",
                    ),
                    color=color,
                )
                for name, label, color in (
                    ("random", "Random", POLICY_COLORS["random"]),
                    ("frontier", "Frontier", POLICY_COLORS["frontier"]),
                    ("trained_mean", "Trained", ACCENT),
                )
            ),
        )
        for metric, title in (
            ("success_rate", "Success"),
            ("completion_progress", "Completion"),
            ("exploration_coverage", "Coverage"),
        )
    )
    _write_text(
        output_directory / "baseline-comparison.svg",
        _panel_chart(
            title="Random, frontier, and trained-mean comparison",
            description="The frontier policy is a handcrafted heuristic, not a learned model.",
            panels=baseline_panels,
            source_hash=source_hash,
        ),
    )

    gaps = _mapping(data["generalization_gaps"], "figure_data.generalization_gaps")
    gap_panels = tuple(
        Panel(
            title=title,
            unit="delta",
            bars=tuple(
                Bar(
                    label=f"Seed {seed}",
                    value=_float(
                        _nested(gaps, f"seed_{seed}", metric),
                        f"figure_data.gaps.seed_{seed}.{metric}",
                    ),
                    color="#dc6b2f" if seed else "#2563eb",
                )
                for seed in range(3)
            ),
        )
        for metric, title in (
            ("targets_recovered", "Target gap"),
            ("completion_progress", "Completion gap"),
            ("exploration_coverage", "Coverage gap"),
        )
    )
    _write_text(
        output_directory / "validation-to-heldout-gaps.svg",
        _panel_chart(
            title="Validation-to-held-out generalization gaps",
            description=(
                "Gap = validation minus held-out; positive values indicate held-out degradation."
            ),
            panels=gap_panels,
            source_hash=source_hash,
        ),
    )

    manifest = {
        "schema_version": 1,
        "source": "docs/day8-final-heldout-results.json",
        "source_sha256": source_hash,
        "provenance": (
            "Values are selected directly from the committed Day 8 JSON; no evaluation is run."
        ),
        "figures": [
            "success-rate-by-policy.svg",
            "targets-recovered-by-policy.svg",
            "exploration-coverage-by-policy.svg",
            "trained-minus-untrained.svg",
            "baseline-comparison.svg",
            "validation-to-heldout-gaps.svg",
        ],
        **data,
    }
    _write_text(
        output_directory / "manifest.json",
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = _parse_args()
    try:
        manifest = generate_figures(args.report.resolve(), args.output_dir.resolve())
    except ValueError as exc:
        raise SystemExit(f"figure generation failed: {exc}") from exc
    sys.stdout.write(
        f"Generated {len(manifest['figures'])} deterministic figures from "
        f"{manifest['source']} ({manifest['source_sha256']}).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
