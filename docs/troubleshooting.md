# Troubleshooting

KOVARA-9 supports Python `>=3.12,<3.13`. Use Python 3.12 for every documented workflow; do not bypass
the package's supported range.

## Python 3.13 is rejected

This is expected. Create a 3.12 environment explicitly:

```console
uv venv --python 3.12
```

Confirm the interpreter before installing:

```console
uv run python --version
```

## The virtual environment has no `pip`

`uv` environments do not need a standalone `pip` executable. Install through `uv`:

```console
uv pip install -e .
```

For the locked contributor environment, use:

```console
uv sync --locked
```

## PowerShell will not activate the environment

Do not weaken a managed execution policy. Run the environment's executable directly:

```powershell
.\.venv\Scripts\kovara9.exe demo
```

For repository development after `uv sync --locked`, this is also valid:

```powershell
uv run kovara9 demo
```

## `kovara9` is not recognized

First verify that the editable package is installed in the intended environment:

```console
uv pip install --reinstall -e .
uv run kovara9 --help
```

On Windows, you can verify the exact executable with:

```powershell
Test-Path .\.venv\Scripts\kovara9.exe
```

On macOS or Linux, use `./.venv/bin/kovara9 --help` if the shell has not refreshed its command path.

## The demo refuses to use the output directory

The demo intentionally refuses to overwrite an existing output path. Choose a new directory:

```console
kovara9 demo --output-dir runs/demo-review-2
```

Inspect or archive the old directory before removing it yourself. There is no force-overwrite flag.

## Validate without writing artifacts

Use the validation-only mode to check the bundled configuration, seeds, and output plan:

```console
kovara9 demo --validate-only
```

Use `--no-render` for the same simulator episodes without terminal frames:

```console
kovara9 demo --no-render
```

## A custom demo configuration is rejected

The configuration is strict by design. It must include both random and frontier examples, unique
names and seeds, declared disjoint seed partitions, a bounded frame limit, and demo seeds within the
training partition. Start from
[`src/kovara9/resources/open_source_demo.yaml`](../src/kovara9/resources/open_source_demo.yaml).

## Still blocked?

Read the [support guide](../.github/SUPPORT.md), search existing issues, and open the matching issue
form. Include the OS, Python and KOVARA-9 versions, installation method, exact command, configuration,
seed, expected behavior, actual behavior, and sanitized logs.
