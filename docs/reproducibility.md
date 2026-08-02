# Reproducibility guide

This guide separates safe presentation workflows from training reproduction and from inspection of
the locked final result. None of the commands below bypasses the consumed-test guard.

## Supported environment

- Python: `>=3.12,<3.13`; v0.1 was evaluated with CPython 3.12.13.
- Dependency manager: [`uv`](https://docs.astral.sh/uv/).
- Reference device: CPU.
- Reference platform: Windows 11; the Python CLI is platform-neutral.
- Lockfile: `uv.lock` is authoritative for the recorded dependency solution.

Install exactly the locked development environment:

```console
uv sync --locked
uv run python --version
uv run kovara9 --help
```

Do not commit `.venv`, caches, run directories, checkpoints, coverage data, or build outputs.

## Determinism boundary

KOVARA-9 derives independent BLAKE2b-based streams from each root seed for actor and critic
initialization, policy sampling, optimizer shuffling, per-environment episode generation, and
validation. It does not use Python's process-randomized `hash()` or a module-global random generator.

`deterministic_torch: true` enables deterministic PyTorch behavior supported by the selected device.
Evaluation uses masked argmax and does not consume actor RNG. CPU is the scientific reference path.
CUDA is optional when a compatible PyTorch installation is present, but exact cross-device or
cross-version bit equivalence is not claimed.

Simulator replay is deterministic for a fixed resolved configuration and seed. Timing metrics are
not deterministic and are reported only as platform diagnostics.

## Configuration and checkpoint identity

Configuration fingerprints are canonical hashes of validated content, not raw path strings.
Training checkpoints bind:

- training, environment, and validation fingerprints;
- observation, state, action, and agent-order signatures;
- actor and critic tensors;
- Adam optimizer slots;
- environment-step, update, episode, and metric history;
- policy and optimizer RNG states;
- collector environments at the rollout boundary.

Resume rejects incompatible configuration, corrupted history, non-finite values, or signature
mismatches before restoring state. Checkpoints are atomically published only after a complete
rollout/update boundary. Verify a frozen candidate without running an episode:

```console
uv run kovara9 config verify-candidate --candidate configs/training/mappo_final_candidate.yaml --freeze-record configs/training/mappo_final_candidate.freeze.json
```

The expected candidate fingerprint is
`f085babe1905a441fed9c1b9c64076f875a476ab753105ca17ce61d9338ada1c`.

### Configuration inventory

- [`mappo_final_candidate.yaml`](../configs/training/mappo_final_candidate.yaml) is the frozen
  Day 6 candidate used for the held-out conclusion.
- [`mappo_day7_shaped.yaml`](../configs/training/mappo_day7_shaped.yaml) and
  [`grid_rescue_day7_recovery.yaml`](../configs/environments/grid_rescue_day7_recovery.yaml)
  preserve the rejected Day 7 reward experiment. They are historical records, not alternatives
  to the frozen candidate.
- [`mappo_final.yaml`](../configs/training/mappo_final.yaml) is an earlier, larger-budget training
  preset. It was not the selected candidate and was not evaluated on the final held-out suite.

The similar filenames do not imply equivalent evidentiary status; the freeze record identifies
the only final candidate.

## Five-minute smoke reproduction

The guided PowerShell demo uses training-domain demonstration seeds `4242` and `4243`, not final
test seeds:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_recruiter_demo.ps1
```

It validates configuration, renders random and frontier behavior, collects a short untrained
rollout, regenerates figures from the committed final JSON, and displays recorded integrity and
quality evidence. It does not optimize, train, or invoke final evaluation.

Platform-neutral equivalents:

```console
uv run kovara9 config validate configs/environments/grid_rescue_easy.yaml
uv run kovara9 env run --config configs/environments/grid_rescue_easy.yaml --agent random --seed 4242 --render ansi
uv run kovara9 env run --config configs/environments/grid_rescue_easy.yaml --agent frontier --seed 4243 --render ansi
uv run kovara9 rollout-smoke --training-config configs/training/mappo_smoke.yaml --steps 8
uv run python scripts/generate_result_figures.py
```

These outputs are demonstrations and pipeline checks, not benchmarks.

## Short training reproduction

This creates a new smoke run using training and validation partitions only. It is not the frozen
candidate and must not be compared as if it were a final result.

```console
uv run kovara9 train --training-config configs/training/mappo_smoke.yaml --output runs/reproduction/mappo-smoke
```

The configuration performs 64 environment transitions and periodic two-seed validation. Runtime on
a normal CPU is typically well below five minutes, but host load and PyTorch startup vary.

To demonstrate exact initialization without optimization:

```console
uv run kovara9 train --training-config configs/training/mappo_smoke.yaml --output runs/reproduction/initial-only --initialize-only
```

## Full Day 6 training reproduction

The final candidate came from three independent Day 6 control runs. Reproduction must retain the
committed reward, architecture, optimizer, environment, and validation suite:

```powershell
0..2 | ForEach-Object {
    uv run kovara9 day6-run-seed `
        --training-config configs/training/mappo_day6_longer.yaml `
        --root-seed $_ `
        --output "runs/reproduction/day6/seed-$_"
}
```

Recorded Day 6 training runtimes were 338.0, 399.5, and 513.1 seconds on the reference CPU, before
separate comparison work. Reproducing those runs does not authorize candidate reselection or access
to the consumed final test partition.

## Validation workflow

Validation configurations select only the declared validation partition. A trusted local checkpoint
can be inspected using:

```console
uv run kovara9 evaluate-checkpoint --checkpoint <trusted-checkpoint.pt> --env-config configs/environments/grid_rescue_medium.yaml --eval-config configs/evaluation/training_validation_smoke.yaml --output runs/reproduction/checkpoint-validation --device cpu
```

Checkpoint files are not committed. Verify the expected SHA-256 before loading one. Never treat a
smoke or validation score as the final held-out result.

## Inspect final results without rerunning final test

The final-test partition is consumed. Inspect the committed report and regenerate presentation
figures only:

```console
uv run python scripts/generate_result_figures.py
```

Relevant immutable records:

- `docs/day8-final-evaluation-preregistration.json`
- `docs/day8-final-heldout-results.json`
- `configs/evaluation/final_test_consumed.json`
- `docs/assets/results/manifest.json`

The figure manifest records both the Day 8 source hash and each generated SVG's SHA-256, allowing
byte-level reproduction checks without evaluating a policy.

Ordinary evaluation and tuning commands reject the consumed test partition. A future replication
requires a separately approved untouched partition and must be labeled replication, not tuning.

## Dependency locking and package build

Quality and packaging commands:

```console
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

If the project `dist` directory is locked, build to a fresh external or OS-temporary directory:

```powershell
$buildRoot = Join-Path $env:TEMP ("kovara9-build-" + [DateTime]::UtcNow.ToString("yyyyMMddHHmmss"))
New-Item -ItemType Directory -Force $buildRoot | Out-Null
uv build --out-dir $buildRoot
```

Install-smoke the resulting wheel into an isolated temporary target or virtual environment; do not
copy it back into the repository.

## Writable temporary directories on Windows

Some managed shells cannot write the configured shared `uv` cache or the default pytest temporary
root. Use task-specific paths rather than changing dependency or test semantics:

```powershell
$env:UV_CACHE_DIR = Join-Path $env:TEMP "kovara9-uv-cache"
$pytestRoot = Join-Path $env:TEMP ("kovara9-pytest-" + [DateTime]::UtcNow.ToString("yyyyMMddHHmmss"))
New-Item -ItemType Directory -Force $pytestRoot | Out-Null
uv run pytest --basetemp $pytestRoot
```

The path workaround is operational only. It must not change seeds, configuration, or scientific
interpretation.

## Known platform limitations

- Exact numeric equivalence is scoped to the recorded software, device, and deterministic-operation
  boundary.
- Windows command examples use PowerShell; the Python CLI itself does not depend on PowerShell.
- ANSI rendering depends on terminal Unicode and color support.
- Inference latency varies with CPU, scheduler, warm-up, and host load.
- The v0.1 artifact writer did not serialize structural-suite inference timing.
- Git credentials and external build/temp locations may be unavailable inside managed sandboxes.
- PyTorch checkpoints must be considered trusted binary inputs.

For methodology details, see [experiment_methodology.md](experiment_methodology.md). For limitations
and proposed controlled extensions, see
[v0.1-limitations-and-future-work.md](v0.1-limitations-and-future-work.md).
