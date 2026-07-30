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
