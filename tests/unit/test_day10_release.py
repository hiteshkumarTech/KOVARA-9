from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from kovara9.config.loader import load_evaluation_config
from kovara9.core.errors import ConfigurationError
from kovara9.experiments.day6 import (
    candidate_configuration_fingerprint,
    load_candidate_freeze,
    validate_candidate_freeze,
)
from kovara9.experiments.day8 import reject_consumed_test_partition

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPOSITORY_ROOT / "docs/day8-final-heldout-results.json"
AUDIT_PATH = REPOSITORY_ROOT / "docs/day10-final-audit.json"
FREEZE_PATH = REPOSITORY_ROOT / "configs/training/mappo_final_candidate.freeze.json"
CANDIDATE_PATH = REPOSITORY_ROOT / "configs/training/mappo_final_candidate.yaml"
CONSUMED_PATH = REPOSITORY_ROOT / "configs/evaluation/final_test_consumed.json"
PREREGISTRATION_PATH = REPOSITORY_ROOT / "docs/day8-final-evaluation-preregistration.json"
MANIFEST_PATH = REPOSITORY_ROOT / "docs/assets/results/manifest.json"


def _json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_row(document: Path, label: str) -> list[str]:
    prefix = f"| {label} |"
    matching = [
        line
        for line in document.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    assert len(matching) == 1
    return [cell.strip() for cell in matching[0].strip("|").split("|")]


def _result_rows(report: dict[str, Any]) -> dict[str, tuple[int, int, float, float, float, float]]:
    policies = report["policy_results"]
    aggregates = report["training_seed_aggregates"]
    random = policies["random"]["pooled"]
    frontier = policies["frontier"]["pooled"]
    return {
        "random": (
            random["successful_episode_length"]["count"],
            200,
            random["success_rate"],
            random["targets_recovered"],
            random["completion_progress"],
            random["exploration_coverage"],
        ),
        "frontier": (
            frontier["successful_episode_length"]["count"],
            200,
            frontier["success_rate"],
            frontier["targets_recovered"],
            frontier["completion_progress"],
            frontier["exploration_coverage"],
        ),
        "untrained": (
            0,
            600,
            aggregates["untrained"]["success_rate_mean"],
            aggregates["untrained"]["targets_recovered_mean"],
            aggregates["untrained"]["completion_progress_mean"],
            aggregates["untrained"]["exploration_coverage_mean"],
        ),
        "trained": (
            0,
            600,
            aggregates["trained"]["success_rate"]["mean"],
            aggregates["trained"]["targets_recovered"]["mean"],
            aggregates["trained"]["completion_progress"]["mean"],
            aggregates["trained"]["exploration_coverage"]["mean"],
        ),
    }


def test_final_audit_fingerprints_match_current_scientific_records() -> None:
    audit = _json(AUDIT_PATH)
    report = _json(REPORT_PATH)
    consumed = _json(CONSUMED_PATH)
    freeze = load_candidate_freeze(FREEZE_PATH)
    fingerprints = audit["fingerprints"]

    validate_candidate_freeze(CANDIDATE_PATH, freeze)
    assert (
        candidate_configuration_fingerprint(CANDIDATE_PATH)
        == fingerprints["candidate_configuration"]
    )
    assert fingerprints["candidate_configuration"] == report["fingerprints"]["candidate"]
    assert fingerprints["candidate_configuration"] == consumed["candidate_fingerprint"]
    assert fingerprints["candidate_configuration"] == freeze.configuration_fingerprint
    assert (
        fingerprints["candidate_fingerprint_recorded_by_freeze"] == freeze.configuration_fingerprint
    )
    assert fingerprints["reward"] == report["fingerprints"]["reward"]
    assert fingerprints["reward"] == freeze.reward_fingerprint
    assert fingerprints["reference_environment"] == report["fingerprints"]["reference_environment"]
    assert fingerprints["reference_environment"] == freeze.environment_fingerprint
    assert fingerprints["reference_environment"] == consumed["environments"]["reference"]
    assert (
        fingerprints["structural_environment"] == report["fingerprints"]["structural_environment"]
    )
    assert fingerprints["structural_environment"] == consumed["environments"]["structural"]
    assert fingerprints["candidate_configuration_file_sha256"] == _sha256(CANDIDATE_PATH)
    assert fingerprints["candidate_freeze_record_sha256"] == _sha256(FREEZE_PATH)
    assert fingerprints["day8_report_sha256"] == _sha256(REPORT_PATH)
    assert fingerprints["preregistration_sha256"] == _sha256(PREREGISTRATION_PATH)
    assert fingerprints["consumption_record_sha256"] == _sha256(CONSUMED_PATH)
    assert fingerprints["dependency_lock_sha256"] == _sha256(REPOSITORY_ROOT / "uv.lock")


def test_final_audit_metrics_and_partitions_match_day8_json() -> None:
    audit = _json(AUDIT_PATH)
    report = _json(REPORT_PATH)
    rows = _result_rows(report)

    assert audit["classification"] == report["classification"]
    assert audit["partitions"]["training_seeds"] == report["partitions"]["training_seeds"]
    for partition in ("validation_seeds", "final_test_seeds"):
        audit_partition = audit["partitions"][partition]
        report_partition = report["partitions"][
            "validation" if partition == "validation_seeds" else "final_test"
        ]
        assert audit_partition["start"] == report_partition["start"]
        assert audit_partition["stop_exclusive"] == report_partition["stop_exclusive"]
        assert audit_partition["count"] == report_partition["count"]
    assert audit["partitions"]["pairwise_disjoint"] is report["partitions"]["pairwise_disjoint"]

    for audit_key, result_key in (
        ("random", "random"),
        ("frontier", "frontier"),
        ("exact_untrained_mean", "untrained"),
        ("frozen_trained_mean", "trained"),
    ):
        successes, episodes, success, targets, completion, coverage = rows[result_key]
        audited = audit["results"][audit_key]
        assert audited["successes"] == successes
        assert audited["episodes"] == episodes
        assert audited["success_rate"] == pytest.approx(success)
        assert audited["targets_recovered"] == pytest.approx(targets)
        assert audited["completion_progress"] == pytest.approx(completion)
        assert audited["exploration_coverage"] == pytest.approx(coverage)


def test_readme_visible_metrics_match_day8_json() -> None:
    rows = _result_rows(_json(REPORT_PATH))
    labels = {
        "random": "Random",
        "frontier": "Frontier heuristic",
        "untrained": "Exact untrained actors, mean",
        "trained": "Frozen trained actors, mean",
    }
    for key, label in labels.items():
        successes, episodes, success, targets, completion, coverage = rows[key]
        cells = _table_row(REPOSITORY_ROOT / "README.md", label)
        assert int(cells[1]) == episodes
        assert int(cells[2]) == successes
        assert float(cells[3]) == pytest.approx(success, abs=0.0005)
        assert float(cells[4]) == pytest.approx(targets, abs=0.00005)
        assert float(cells[5]) == pytest.approx(completion, abs=0.00005)
        assert float(cells[6]) == pytest.approx(coverage, abs=0.00005)


@pytest.mark.parametrize(
    ("relative_path", "labels", "include_successes"),
    [
        (
            "MODEL_CARD.md",
            {
                "random": "Random",
                "frontier": "Handcrafted frontier",
                "untrained": "Exact untrained mean",
                "trained": "Frozen trained mean",
            },
            False,
        ),
        (
            "docs/research-summary.md",
            {
                "random": "Random",
                "frontier": "Frontier heuristic",
                "untrained": "Exact untrained, three-seed mean",
                "trained": "Frozen trained, three-seed mean",
            },
            True,
        ),
    ],
)
def test_scientific_summary_visible_metrics_match_day8_json(
    relative_path: str,
    labels: dict[str, str],
    *,
    include_successes: bool,
) -> None:
    rows = _result_rows(_json(REPORT_PATH))
    document = REPOSITORY_ROOT / relative_path
    for key, label in labels.items():
        successes, episodes, success, targets, completion, coverage = rows[key]
        cells = _table_row(document, label)
        expected_episode_column = (
            f"{successes} / {episodes}" if include_successes else str(episodes)
        )
        assert cells[1] == expected_episode_column
        assert float(cells[2]) == pytest.approx(success, abs=0.0005)
        assert float(cells[3]) == pytest.approx(targets, abs=0.00005)
        assert float(cells[4]) == pytest.approx(completion, abs=0.00005)
        assert float(cells[5]) == pytest.approx(coverage, abs=0.00005)


def test_figure_provenance_and_hashes_are_stable() -> None:
    manifest = _json(MANIFEST_PATH)
    script = (REPOSITORY_ROOT / "scripts/generate_result_figures.py").read_text(encoding="utf-8")
    assert manifest["source_sha256"] == _sha256(REPORT_PATH)
    assert manifest["schema_version"] == 2
    assert set(manifest["figure_sha256"]) == set(manifest["figures"])
    for filename, expected_hash in manifest["figure_sha256"].items():
        assert _sha256(MANIFEST_PATH.parent / filename) == expected_hash
    for forbidden_literal in ("0.855", "0.6133333333333334", "0.5016276777306968"):
        assert forbidden_literal not in script


def test_consumed_final_partition_guard_remains_active() -> None:
    consumed = _json(CONSUMED_PATH)
    evaluation = load_evaluation_config(REPOSITORY_ROOT / "configs/evaluation/generalization.yaml")
    assert consumed["status"] == "complete"
    with pytest.raises(ConfigurationError, match="partition is consumed"):
        reject_consumed_test_partition(evaluation, CONSUMED_PATH)


def test_release_version_matches_package_and_citation() -> None:
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    audit = _json(AUDIT_PATH)
    version = project["project"]["version"]
    assert version == "0.1.0"
    assert citation["version"] == version
    assert audit["software"]["package_version"] == version
    assert "## [0.1.0]" in (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "`v0.1.0`" in (REPOSITORY_ROOT / "docs/github-release-v0.1.0.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/github-pr-description.md",
        "docs/github-release-v0.1.0.md",
        "docs/final-portfolio-wording.md",
    ],
)
def test_release_documents_preserve_the_honest_result(relative_path: str) -> None:
    document = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8").casefold()
    normalized = re.sub(r"\s+", " ", document)
    assert "exploration transfer without task completion" in normalized
    assert any(pattern in normalized for pattern in ("0 of 600", "0/600", "0 successes in 600"))
    assert "random" in normalized
    assert "frontier" in normalized


def test_portfolio_wording_does_not_claim_successful_task_completion() -> None:
    wording = (REPOSITORY_ROOT / "docs/final-portfolio-wording.md").read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", wording.casefold())
    assert (
        "failure to achieve full task completion" in normalized
        or "without task completion" in normalized
    )
    assert "not a claim of a successful rescue agent" in normalized
    assert "random succeeded 28" in normalized
    assert "frontier" in normalized
    assert "171" in normalized


def test_old_repository_url_is_allowed_only_until_rename_is_confirmed() -> None:
    audit = _json(AUDIT_PATH)
    old_url = "github.com/hiteshkumarTech/KOVARA-9-"
    candidates = [
        *REPOSITORY_ROOT.glob("*.md"),
        REPOSITORY_ROOT / "CITATION.cff",
        REPOSITORY_ROOT / "pyproject.toml",
        *(REPOSITORY_ROOT / "docs").rglob("*.md"),
        *(REPOSITORY_ROOT / "docs").rglob("*.json"),
        *(REPOSITORY_ROOT / ".github").rglob("*.yml"),
    ]
    tracked_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in candidates
        if path.name != "repository-rename-checklist.md"
    )
    if audit["repository_rename"]["confirmed"]:
        assert old_url not in tracked_text
    else:
        assert audit["repository_rename"]["remote_changed_by_audit"] is False
        assert "The rename has **not** been confirmed" in (
            REPOSITORY_ROOT / "docs/repository-rename-checklist.md"
        ).read_text(encoding="utf-8")
