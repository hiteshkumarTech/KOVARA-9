# Contributing

Use a focused branch and keep changes small enough to review. Every behavioral change must
include tests and update affected documentation or configuration examples.

Before submitting a change, run:

```console
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Do not commit generated runs, environments, secrets, credentials, model checkpoints, or
claims unsupported by recorded experiments. Architectural changes must record alternatives,
rationale, and trade-offs in `docs/decisions.md`.

Contributions are accepted under the Apache License 2.0.
