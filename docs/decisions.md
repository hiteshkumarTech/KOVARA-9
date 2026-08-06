# Architecture decision log

## ADR-001: CPU-first grid simulator

**Decision:** begin with a NumPy grid rather than a 3D or physics engine.

**Rationale:** deterministic transitions, property testing, and rapid experiments are more valuable
than visual fidelity at this stage. The platform must run on integrated graphics and limited RAM.

**Trade-off:** continuous control and perception are postponed; renderer and snapshot boundaries
preserve a migration path.

## ADR-002: PettingZoo Parallel API

**Decision:** expose simultaneous multi-agent steps through `ParallelEnv`.

**Rationale:** it matches the environment clock, provides standard spaces and compliance tests, and
avoids a proprietary trainer coupling.

**Trade-off:** PettingZoo is a core dependency, and collision semantics still require explicit
project documentation.

## ADR-003: uv and a committed lockfile

**Decision:** use uv, a project-local environment, and `uv.lock`.

**Rationale:** the installed machine already has uv, interpreter selection is ambiguous, and exact
dependency resolution is necessary for reproducibility.

**Trade-off:** contributors must install uv. A repository-local cache is used because the current
global cache is inaccessible.

## ADR-004: local transparent artifacts

**Decision:** record resolved YAML, JSON manifests, episode JSONL, and aggregate JSON.

**Rationale:** these formats are inspectable, portable, scriptable, and require no service.

**Trade-off:** remote tracking and dashboards are unavailable until an evidence-based need emerges.

## ADR-005: opaque budgeted broadcast tokens

**Decision:** messages are small, global, one-step, opaque tokens with independent agent budgets.

**Rationale:** bandwidth and cost are measurable without hard-coding language or adding spatial
delivery complexity to the first simulator.

**Trade-off:** global delivery is less physically grounded than range-limited communication.

## ADR-006: stable communication actions with explicit rejection

**Decision:** keep the dictionary action space fixed and expose separate fixed-shape movement and
message masks. Over-budget non-silent messages become reported rejected no-ops.

**Rationale:** changing spaces or throwing on a sampled action violates the Parallel API contract,
while silently accepting or dropping a token corrupts communication metrics.

**Trade-off:** policies may still ignore the mask, so evaluators must distinguish attempted,
rejected, and accepted communication through personal info and environment event records.

## ADR-007: evaluation configuration owns structural comparisons

**Decision:** a comparison evaluation file owns both environment paths and the complete seed
partition declaration. Content fingerprints, not filenames, establish that environments differ.

**Rationale:** a duplicate CLI reference environment and path-only checks permit accidental or
mislabelled comparisons.

**Trade-off:** comparison runs use a distinct CLI form without `--env-config`, and portable relative
paths are interpreted from the evaluation file rather than the launch directory.

## ADR-008: one feed-forward parameter-sharing CTDE learner

**Decision:** v0.1 uses PyTorch, one shared feed-forward actor, one centralized team-value critic,
and a single MAPPO-style PPO training path.

**Rationale:** this directly tests the approved parameter-sharing research question while keeping
decentralized execution and centralized training mechanically separate and reviewable.

**Trade-off:** the initial actor has no recurrence and the critic estimates one shared team value.
This is a documented MAPPO-style simplification, not a claim of reproducing every MAPPO variant.

## ADR-009: exact rollout-boundary checkpoints

**Decision:** checkpoint only after a complete rollout and PPO update, and persist the in-progress
simulator states together with model, optimizer, counters, metric history, and explicit policy and
minibatch RNG streams.

**Rationale:** resetting environments or restoring only model weights changes the next on-policy
rollout whenever a checkpoint falls inside an episode. Validated environment-state adapters preserve
exact continuation without coupling the learner to grid-rescue transition rules or replaying an
unbounded action history.

**Trade-off:** checkpoints contain training-only critic and simulator state in addition to the actor.
Saved-policy evaluation loads the actor separately and validates only decentralized
observation/action compatibility.

## ADR-010: packaged baseline walkthrough

**Decision:** expose a fast `kovara9 demo` command backed by a validated YAML resource inside the
wheel. It runs only the existing random and handcrafted frontier policies on explicit training-domain
seeds. ANSI frames consume defensive snapshots through an evaluation observer.

**Rationale:** the v0.1 evidence tour is PowerShell-first and repository-relative. A prospective
contributor should be able to verify real simulator behavior from any supported terminal or from an
installed wheel without a checkpoint, private artifact, training run, or final-test access.

**Trade-off:** two fixed episodes are illustrative examples, not baseline estimates. The longer
PowerShell evidence tour remains the workflow for freeze checks, frozen-figure regeneration, and
presentation-integrity tests.
