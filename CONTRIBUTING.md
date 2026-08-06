# Contributing

KOVARA-9 welcomes focused fixes, reproducibility improvements, documentation corrections, and
well-scoped adapter proposals. Read the [code of conduct](CODE_OF_CONDUCT.md) before participating.

## Find the right starting point

- Reproduce a bug with the smallest validated configuration before proposing a fix.
- Use the reproducibility issue form for fingerprint, seed, artifact, or documented-result mismatches.
- Discuss an external environment through the environment proposal form before implementation.
- Tie feature requests to a concrete research or developer workflow and state explicit non-goals.

Look for issues labeled by the maintainers as suitable for a first contribution. If no such issue is
available, documentation links, error context, and narrowly scoped tests are good places to help.

## Development setup

Use Python `>=3.12,<3.13` and the committed lockfile.

```console
uv venv --python 3.12
uv sync --locked
uv run kovara9 demo --validate-only
```

On Windows, `uv run ...` works without activating the environment. See
[troubleshooting](docs/troubleshooting.md) for shell and installation help.

Create a focused branch and keep changes small enough to review. Every behavioral change must include
tests and update affected documentation or configuration examples.

## Repository boundaries

Do not rewrite the v0.1 candidate, preregistration, consumed-test lock, held-out reports, or result
assets. New research must use new versioned configurations and an untouched evaluation partition.

All changes must preserve these boundaries:

- simulator transitions are independent from rendering and user interfaces;
- policies and algorithms depend on protocols, not concrete environments;
- decentralized policies consume only their own agent observation;
- research parameters live in validated configuration;
- stochastic behavior uses explicit reproducible seed streams; and
- metrics, checkpoints, integrations, and completed functionality are never fabricated.

Demo changes must keep rendering downstream of defensive simulator snapshots. Architecture,
methodology, metric, reward, or reproducibility changes require updated documentation and a recorded
decision with alternatives and trade-offs in [`docs/decisions.md`](docs/decisions.md).

## Quality gates

Before submitting a pull request, run:

```console
uv sync --locked
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
uv build
uv run kovara9 --help
uv run kovara9 demo --validate-only
uv run kovara9 demo --no-render
```

Run the full coverage command documented in [`pyproject.toml`](pyproject.toml) when checking the
repository threshold. Do not lower thresholds to make a change pass. Use the exact built wheel for an
installed-wheel smoke check when the CLI, packaging, or dependencies change.

Do not commit generated runs, virtual environments, caches, secrets, credentials, private datasets,
model checkpoints, or claims unsupported by recorded experiments.

## Pull requests

Complete the pull-request template, link the relevant issue, list exact commands and outcomes, and
call out known limitations. A maintainer may request a smaller scope or a decision record before
reviewing an architectural change.

Contributions are accepted under the [Apache License 2.0](LICENSE).
