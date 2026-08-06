from __future__ import annotations

import json
import sys
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIRECTORY = REPOSITORY_ROOT / "docs" / "assets"
ASSET_NAMES = (
    "demo-frame.png",
    "demo-frontier.gif",
    "policy-comparison.png",
    "kovara9-social-preview.png",
    "readme-assets-manifest.json",
)


def _load_generator() -> Callable[[Path], dict[str, object]]:
    script_path = REPOSITORY_ROOT / "scripts" / "generate_readme_assets.py"
    specification = spec_from_file_location("kovara9_readme_asset_script", script_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load README asset generator: {script_path}")
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return cast(Callable[[Path], dict[str, object]], module.generate_assets)


def test_readme_asset_generation_is_byte_deterministic(tmp_path: Path) -> None:
    manifest = _load_generator()(tmp_path)
    assert manifest["demo"] == {
        "policy": "frontier",
        "label": "handcrafted-frontier-example",
        "seed": 4243,
        "seed_partition": "train/development",
        "final_test_seed_used": False,
        "steps": 9,
        "targets_recovered": 2,
        "target_count": 2,
        "success": True,
        "frames": 10,
        "purpose": "behavioral walkthrough; not a benchmark estimate",
    }
    for name in ASSET_NAMES:
        assert (tmp_path / name).read_bytes() == (ASSET_DIRECTORY / name).read_bytes()


def test_readme_asset_manifest_has_verified_sources() -> None:
    manifest: dict[str, Any] = json.loads(
        (ASSET_DIRECTORY / "readme-assets-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["comparison"] == {
        "classification": "exploration_transfer_without_task_completion",
        "purpose": "derived visualization of frozen v0.1.0 held-out results",
        "success_rates": {
            "frontier": 0.855,
            "random": 0.14,
            "trained_mean": 0.0,
            "untrained_mean": 0.0,
        },
    }
    assert manifest["sources"]["frozen_results"] == "docs/day8-final-heldout-results.json"


def test_brand_svgs_are_valid_xml_with_accessible_titles() -> None:
    for name in ("kovara9-logo.svg", "kovara9-banner.svg"):
        root = ElementTree.parse(ASSET_DIRECTORY / name).getroot()
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        assert root.find("svg:title", namespace) is not None
        assert root.find("svg:desc", namespace) is not None


def test_github_issue_forms_are_valid_yaml() -> None:
    forms = REPOSITORY_ROOT / ".github" / "ISSUE_TEMPLATE"
    for path in sorted(forms.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict)
        if path.name != "config.yml":
            assert isinstance(document.get("body"), list)
            assert document["body"]
