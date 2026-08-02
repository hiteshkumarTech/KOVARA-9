from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from kovara9.config.models import EvaluationConfig, SeedPartitionsConfig, SeedRangeConfig
from kovara9.core.errors import ConfigurationError
from kovara9.experiments.day8 import reject_consumed_test_partition

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_figure_script() -> tuple[tuple[str, ...], Callable[[Path, Path], dict[str, Any]]]:
    script_path = REPOSITORY_ROOT / "scripts/generate_result_figures.py"
    spec = spec_from_file_location("kovara9_day9_figure_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load figure script: {script_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return (
        cast(tuple[str, ...], module.POLICY_ORDER),
        cast(Callable[[Path, Path], dict[str, Any]], module.generate_figures),
    )


POLICY_ORDER, generate_figures = _load_figure_script()
REPORT_PATH = REPOSITORY_ROOT / "docs/day8-final-heldout-results.json"
MANIFEST_PATH = REPOSITORY_ROOT / "docs/assets/results/manifest.json"
DEMO_PATH = REPOSITORY_ROOT / "scripts/run_recruiter_demo.ps1"
METRIC_MARKER = re.compile(r"<!-- day8-metric:([a-z0-9_]+)=([^ ]+) -->")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
DAY9_DOCUMENTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "MODEL_CARD.md",
    REPOSITORY_ROOT / "docs/architecture.md",
    REPOSITORY_ROOT / "docs/recruiter-demo.md",
    REPOSITORY_ROOT / "docs/reproducibility.md",
    REPOSITORY_ROOT / "docs/research-summary.md",
    REPOSITORY_ROOT / "docs/v0.1-limitations-and-future-work.md",
)


def _json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _local_links(document: Path) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for match in MARKDOWN_LINK.finditer(document.read_text(encoding="utf-8")):
        target = match.group(1).strip().strip("<>").split(maxsplit=1)[0]
        target = target.split("#", maxsplit=1)[0]
        if not target or "://" in target or target.startswith(("mailto:", "#")):
            continue
        resolved.append((document.parent / target).resolve())
    return tuple(resolved)


def _metric_markers(path: Path) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in METRIC_MARKER.findall(path.read_text(encoding="utf-8"))
    }


def _expected_document_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        "trained_success_rate": report["training_seed_aggregates"]["trained"]["success_rate"][
            "mean"
        ],
        "random_success_rate": report["policy_results"]["random"]["pooled"]["success_rate"],
        "frontier_success_rate": report["policy_results"]["frontier"]["pooled"]["success_rate"],
        "trained_targets_recovered": report["training_seed_aggregates"]["trained"][
            "targets_recovered"
        ]["mean"],
        "trained_completion_progress": report["training_seed_aggregates"]["trained"][
            "completion_progress"
        ]["mean"],
        "trained_exploration_coverage": report["training_seed_aggregates"]["trained"][
            "exploration_coverage"
        ]["mean"],
    }


def _powershell() -> str:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is unavailable on this platform")
    return executable


def test_readme_links_point_to_existing_repository_files() -> None:
    links = _local_links(REPOSITORY_ROOT / "README.md")
    assert links
    assert all(path.exists() for path in links)


def test_day9_documentation_links_resolve() -> None:
    missing = [
        (document.relative_to(REPOSITORY_ROOT), target)
        for document in DAY9_DOCUMENTS
        for target in _local_links(document)
        if not target.exists()
    ]
    assert missing == []


def test_readme_leads_with_the_honest_result_statement() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", readme)
    framing = (
        "A reproducible multi-agent reinforcement-learning research platform that demonstrated "
        "partial exploration transfer but failed to learn full cooperative task completion under "
        "the tested configuration."
    )
    required = (
        "The trained policies improved exploration and partial target recovery over their exact "
        "untrained initializations, but achieved zero full-task successes on the held-out "
        "evaluation and remained below random and frontier baselines."
    )
    assert framing in normalized
    assert required in normalized
    assert normalized.index(framing) < normalized.index("## Research question")
    assert normalized.index(required) < normalized.index("## Research question")


def test_result_figure_values_match_the_final_report_json() -> None:
    report = _json(REPORT_PATH)
    manifest = _json(MANIFEST_PATH)
    assert manifest["source_sha256"] == hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()
    for policy in POLICY_ORDER:
        for metric in ("success_rate", "targets_recovered", "exploration_coverage"):
            assert manifest["policy_metrics"][policy][metric] == pytest.approx(
                report["policy_results"][policy]["pooled"][metric]
            )
    for seed in range(3):
        for metric in ("targets_recovered", "exploration_coverage"):
            assert manifest["paired_trained_minus_untrained"][f"seed_{seed}"][
                metric
            ] == pytest.approx(report["paired_trained_minus_untrained"][f"seed_{seed}"][metric])


def test_generated_policy_figures_contain_every_required_policy() -> None:
    manifest = _json(MANIFEST_PATH)
    assert manifest["policy_order"] == list(POLICY_ORDER)
    for filename in (
        "success-rate-by-policy.svg",
        "targets-recovered-by-policy.svg",
        "exploration-coverage-by-policy.svg",
    ):
        svg = (MANIFEST_PATH.parent / filename).read_text(encoding="utf-8")
        for policy in POLICY_ORDER:
            assert f'data-policy="{policy}"' in svg


def test_result_figure_generation_is_deterministic(tmp_path: Path) -> None:
    manifest = generate_figures(REPORT_PATH, tmp_path)
    assert manifest["figures"] == _json(MANIFEST_PATH)["figures"]
    for filename in manifest["figures"]:
        assert (tmp_path / filename).read_bytes() == (MANIFEST_PATH.parent / filename).read_bytes()
    assert (tmp_path / "manifest.json").read_bytes() == MANIFEST_PATH.read_bytes()


def test_demo_seeds_do_not_overlap_final_test_partition() -> None:
    script = DEMO_PATH.read_text(encoding="utf-8")
    seed_match = re.search(r"\$DemoSeeds\s*=\s*@\(([^)]+)\)", script)
    assert seed_match is not None
    demo_seeds = {int(value.strip()) for value in seed_match.group(1).split(",")}
    final_partition = _json(REPORT_PATH)["partitions"]["configured"]["test"]
    final_seeds = range(final_partition["start"], final_partition["stop_exclusive"])
    assert demo_seeds.isdisjoint(final_seeds)


def test_recruiter_demo_does_not_invoke_final_test_workflows() -> None:
    script = DEMO_PATH.read_text(encoding="utf-8").casefold()
    for forbidden in ("final-evaluate", "generalization.yaml", "--allow-test-partition"):
        assert forbidden not in script
    assert "training_validation_smoke.yaml" in script


def test_recruiter_demo_validates_without_a_checkpoint() -> None:
    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DEMO_PATH),
            "-ValidateOnly",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DEMO VALIDATION COMPLETE" in result.stdout
    assert "No checkpoint supplied" in result.stdout


def test_recruiter_demo_rejects_a_missing_optional_checkpoint(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    result = subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DEMO_PATH),
            "-CheckpointPath",
            str(missing),
            "-ValidateOnly",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "does not exist or is not a file" in result.stderr


def test_final_consumption_lock_remains_active() -> None:
    consumed = REPOSITORY_ROOT / "configs/evaluation/final_test_consumed.json"
    assert _json(consumed)["status"] == "complete"
    synthetic_test_suite = EvaluationConfig(
        name="day9-synthetic-lock-check",
        seeds=(77,),
        seed_partition="test",
        seed_partitions=SeedPartitionsConfig(
            train=SeedRangeConfig(start=0, count=10),
            validation=SeedRangeConfig(start=20, count=10),
            test=SeedRangeConfig(start=77, count=1),
        ),
        bootstrap_samples=0,
        bootstrap_confidence=0.95,
    )
    with pytest.raises(ConfigurationError, match="partition is consumed"):
        reject_consumed_test_partition(synthetic_test_suite, consumed)


@pytest.mark.parametrize(
    "document", [REPOSITORY_ROOT / "MODEL_CARD.md", REPOSITORY_ROOT / "docs/research-summary.md"]
)
def test_documented_metrics_match_final_json(document: Path) -> None:
    assert _metric_markers(document) == pytest.approx(
        _expected_document_metrics(_json(REPORT_PATH))
    )


def test_citation_cff_parses_and_makes_no_publication_claim() -> None:
    citation = yaml.safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "Apache-2.0"
    assert citation["authors"] == [{"name": "KOVARA-9 Contributors"}]
    assert "doi" not in citation


def test_no_forbidden_artifact_is_tracked() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    forbidden = re.compile(
        r"^(runs|models|checkpoints|dist|build|\.venv|\.uv-cache|\.pytest_cache|"
        r"\.mypy_cache|\.ruff_cache|\.hypothesis)(/|$)"
    )
    assert [path for path in tracked if forbidden.search(path)] == []
