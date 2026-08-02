# Five-minute recruiter demo

## What this demo proves

In approximately five minutes on CPU, the demo shows that KOVARA-9 has a real procedural
multi-agent environment, distinct random and handcrafted baseline behaviors, a working typed rollout
pipeline, deterministic configuration/fingerprint checks, JSON-derived final-result figures, and
focused presentation-integrity tests.

It does **not** prove task learning or report a new benchmark. The frozen learned policies had zero
full-task successes in the final held-out evaluation.

## Prerequisites

- Python 3.12
- `uv`
- PowerShell 5.1 or newer
- A terminal capable of displaying the ANSI grid

No CUDA device, private credential, raw run directory, or committed checkpoint is required.

## Run it

From the repository root:

```powershell
uv sync --locked
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_recruiter_demo.ps1
```

The script uses demonstration seeds `4242` and `4243`, which belong to the training-domain range and
do not overlap the final-test partition. It validates the completed consumption lock before running
anything.

## Walkthrough

### 1. Configuration validation

The easy procedural environment and smoke training configuration are parsed through the same typed
loaders used by production workflows. Invalid fields or inconsistent paths fail with context.

### 2. Frozen-candidate verification

The CLI recomputes the final candidate, reward, environment, and validation identities. It prints
the canonical candidate fingerprint and confirms that the final partition is consumed. This command
does not load a test episode.

### 3. Random behavior

One fixed-seed random episode runs headlessly and reports its outcome. It is a behavioral example,
not an estimate of random-policy performance.

### 4. Frontier behavior

One fixed-seed frontier episode renders in the terminal. Frontier is a deterministic handcrafted
exploration heuristic; it is not a trained model. Its visible behavior helps distinguish engineered
structure from the learned candidate.

### 5. Rollout smoke

An eight-step untrained rollout exercises local actor encoding, centralized critic encoding, action
masks, explicit seed streams, synchronous collection, and boundary records. It performs no
optimization, saves no checkpoint, and makes no learning claim.

### 6. Existing final-result figures

The figure generator reads only `docs/day8-final-heldout-results.json`, then deterministically
rewrites six lightweight SVGs and a provenance manifest containing their SHA-256 hashes. It does not
open raw episode artifacts or invoke an evaluator.

### 7. Focused presentation-integrity tests

The demo runs the Day 9 tests without repository-wide coverage enforcement. They validate links,
figure provenance, required policies, safe demo seeds, final-command exclusion, optional-checkpoint
validation, metric consistency, citation metadata, and tracked-artifact hygiene. The full repository
suite remains a separate quality gate.

## Optional trusted checkpoint

Trained checkpoints are intentionally excluded from Git. If you possess a trusted local `.pt` file,
verify its SHA-256 against the appropriate experiment record and pass it explicitly:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_recruiter_demo.ps1 `
    -CheckpointPath C:\trusted\best.pt
```

The script validates that the path exists, is a file, and has a `.pt` suffix before running any demo
command. It evaluates the actor deterministically on the two-seed validation smoke suite and writes
temporary output outside the repository. It never selects the consumed final-test suite.

Do not download or load an arbitrary checkpoint merely for the demo. PyTorch checkpoint files must
be treated as trusted binary inputs.

## Fast safety validation

Maintainers can validate inputs and seed/lock safety without launching episodes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_recruiter_demo.ps1 `
    -ValidateOnly
```

This mode is also used by automated tests.

## Expected conclusion

The appropriate closing statement is:

> KOVARA-9 demonstrates a carefully tested, reproducible MARL research workflow and measurable
> transfer of partial exploration behavior. Under the frozen v0.1 configuration, complete
> cooperative task behavior did not emerge, and both random and frontier baselines remained
> stronger on central task metrics.
