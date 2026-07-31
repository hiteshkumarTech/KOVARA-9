# KOVARA-9

KOVARA-9 is a serious multi-agent embodied-AI research platform for studying whether
independently learning agents can develop useful coordination strategies and generalize
those strategies to unseen procedural environments.

Phase 0 provides a deterministic, CPU-friendly cooperative grid simulator, decentralized
baseline policies, evaluation metrics, reproducible artifacts, and strict engineering
foundations. It intentionally contains no learning algorithm, LLM, 3D engine, database,
dashboard, or cloud infrastructure.

## Quick start

Prerequisite: [uv](https://docs.astral.sh/uv/).

```console
uv sync --locked
uv run kovara9 config validate configs/environments/grid_rescue_easy.yaml
uv run kovara9 env run --config configs/environments/grid_rescue_easy.yaml --agent frontier --seed 7 --render ansi
uv run kovara9 evaluate --env-config configs/environments/grid_rescue_easy.yaml --eval-config configs/evaluation/smoke.yaml --agent random --output runs/smoke
uv run kovara9 evaluate --eval-config configs/evaluation/generalization.yaml --agent frontier --output runs/generalization
```

Structural-comparison evaluation files declare both reference and held-out environments; supplying
`--env-config` for such a run is rejected to avoid conflicting experiment definitions.

If the host defines an inaccessible `UV_CACHE_DIR`, override it for the shell before running uv.
For PowerShell:

```powershell
$env:UV_CACHE_DIR = (Join-Path $PWD ".uv-cache")
```

## Quality checks

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Read [the architecture](docs/architecture.md), [experiment methodology](docs/experiment_methodology.md),
and [roadmap](docs/roadmap.md) before extending the platform.
