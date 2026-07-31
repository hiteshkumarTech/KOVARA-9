# KOVARA-9

KOVARA-9 is a serious multi-agent embodied-AI research platform for studying whether
independently learning agents can develop useful coordination strategies and generalize
those strategies to unseen procedural environments.

Phase 0 provides a deterministic, CPU-friendly cooperative grid simulator, decentralized baseline
policies, evaluation metrics, reproducible artifacts, and strict engineering foundations. The v0.1
branch now includes a bounded PyTorch MAPPO-style optimization path: shared decentralized actor,
centralized critic, masked factored actions, synchronous rollout collection, GAE, and clipped PPO
updates. Checkpointing, resume, the full training loop, and learned-policy results are not yet
implemented. The project intentionally excludes LLMs, 3D engines, databases, dashboards, and cloud
infrastructure.

## Quick start

Prerequisite: [uv](https://docs.astral.sh/uv/).

```console
uv sync --locked
uv run kovara9 config validate configs/environments/grid_rescue_easy.yaml
uv run kovara9 env run --config configs/environments/grid_rescue_easy.yaml --agent frontier --seed 7 --render ansi
uv run kovara9 evaluate --env-config configs/environments/grid_rescue_easy.yaml --eval-config configs/evaluation/smoke.yaml --agent random --output runs/smoke
uv run kovara9 evaluate --eval-config configs/evaluation/generalization.yaml --agent frontier --output runs/generalization
uv run kovara9 rollout-smoke --training-config configs/training/mappo_smoke.yaml --steps 8
uv run kovara9 update-smoke --training-config configs/training/mappo_smoke.yaml
```

`update-smoke` proves only that one finite optimization update changes both networks. It is not a
training run, benchmark, checkpoint, or claim that useful behavior has been learned.

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
