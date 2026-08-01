# KOVARA-9

KOVARA-9 is a serious multi-agent embodied-AI research platform for studying whether
independently learning agents can develop useful coordination strategies and generalize
those strategies to unseen procedural environments.

Phase 0 provides a deterministic, CPU-friendly cooperative grid simulator, decentralized baseline
policies, evaluation metrics, reproducible artifacts, and strict engineering foundations. The v0.1
branch now includes a PyTorch MAPPO-style training path: a shared decentralized actor, centralized
critic, masked factored actions, synchronous rollouts, GAE, clipped PPO updates, atomic checkpoints,
exact rollout-boundary resume, and deterministic saved-actor evaluation. No useful learned-policy
result is claimed yet. The project intentionally excludes LLMs, 3D engines, databases, dashboards,
and cloud infrastructure.

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
uv run kovara9 train --training-config configs/training/mappo_day5_short.yaml --output runs/day5/initial --initialize-only
uv run kovara9 train --training-config configs/training/mappo_smoke.yaml --output runs/mappo-smoke
uv run kovara9 evaluate-checkpoint --checkpoint runs/mappo-smoke/checkpoints/step-000000000064.pt --env-config configs/environments/grid_rescue_easy.yaml --eval-config configs/evaluation/training_validation_smoke.yaml --output runs/mappo-smoke-evaluation
```

`update-smoke` proves only that one finite optimization update changes both networks. It is not a
training run, benchmark, checkpoint, or claim that useful behavior has been learned.

`train` writes resolved configuration, provenance, an atomic metrics journal, and scheduled
checkpoints. `--stop-after-environment-steps` creates a deliberately bounded checkpoint at a
rollout-aligned step for interruption/resume verification; resume into a new collision-safe output
directory with `--resume-from`. `--initialize-only` persists the exact untrained actor, critic,
optimizer, collector, and RNG state without performing a rollout or update. `evaluate-checkpoint`
loads only the decentralized actor for masked
deterministic inference. `compare-policies` evaluates random, frontier, untrained neural, and saved
checkpoint policies on the same configured episode seeds and writes aligned per-seed results.

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
