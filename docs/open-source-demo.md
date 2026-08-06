# Open-source demo

## Purpose

`kovara9 demo` is the shortest honest path from installation to real KOVARA-9 behavior. It runs two
baseline-controlled episodes in the procedural PettingZoo environment and normally completes in a
few seconds after Python and PyTorch start. It is available on every supported platform and from an
installed wheel.

This walkthrough is not training, evaluation, or a benchmark. It does not demonstrate that the
learned v0.1 actors solved the task; they recorded zero full-task successes in the frozen held-out
evaluation.

## Run

From a repository checkout:

```console
uv sync --locked
uv run kovara9 demo
```

After installing a built wheel, the equivalent command is simply:

```console
kovara9 demo
```

The default walkthrough runs a random example with seed `4242` without rendering, then renders the
handcrafted frontier example with seed `4243`. On the committed configuration the random example
recovers one of two targets before the time limit and the frontier example recovers both. These are
individual deterministic examples, not estimates of policy performance.

## Configuration and seed safety

The authoritative default is
[`src/kovara9/resources/open_source_demo.yaml`](../src/kovara9/resources/open_source_demo.yaml). It
is included in the wheel and parsed through strict Pydantic models before an environment is created.
The file declares:

- the complete environment and reward configuration;
- mutually disjoint train, validation, and test seed ranges;
- unique episode names and explicit seeds;
- exactly the existing random and frontier baseline policies; and
- whether each episode renders and the maximum captured frame count.

Validation rejects unknown fields, duplicate names or seeds, missing baseline coverage, and any demo
seed outside the declared training partition. To inspect a modified copy without executing it:

```console
uv run kovara9 demo --config path/to/demo.yaml --validate-only
uv run kovara9 config validate path/to/demo.yaml
```

The command has no checkpoint option and no route to `final-evaluate`. Custom demo files remain
illustrative configurations and must not be presented as new benchmark evidence.

## Transparent artifacts

Use a new output directory to save the resolved configuration and deterministic report:

```console
uv run kovara9 demo --no-render --output runs/open-source-demo
```

The command creates `demo.resolved.yaml` and `report.json` and refuses to overwrite an existing
directory. The report records the configuration fingerprint, episode seeds and factual simulator
metrics, plus explicit `false` flags for training, final evaluation, and learned-checkpoint loading.
Rendered frames are excluded so `--render` and `--no-render` produce the same evidence.

## Architecture boundary

The demo reuses `run_episode` and the environment-independent `Policy` protocol. Each random or
frontier instance receives only its own local observation and personal transition outcome. An
optional evaluation observer receives defensive snapshots after reset and each completed transition;
the ANSI renderer consumes those copies and cannot mutate or advance the simulator. Centralized
state is never passed to a policy.

For the longer repository evidence workflow—including frozen-candidate verification, rollout smoke,
frozen-result figure regeneration, and presentation-integrity tests—use the
[five-minute recruiter demo](recruiter-demo.md).
