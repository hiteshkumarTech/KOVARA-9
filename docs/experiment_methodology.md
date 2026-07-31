# Experiment methodology

## Reproducibility

Every run records resolved configuration, exact configured and executed seeds, canonical
configuration fingerprints, package and Python versions, platform information, policy name and
parameters, lockfile hash, and Git repository root, commit, and dirty-tree status when launched from
any directory inside a repository. Environment randomness uses only episode-local NumPy generators.
Artifact creation rejects result records whose ordered seeds differ from the configured suite.
Deterministic simulator replay does not imply deterministic future neural-network training.

Evaluation files declare typed, half-open seed ranges for all three scientific partitions and select
the partition used by that suite. This required metadata is evaluation schema version 2. The
committed ranges are:

- training domain: `0–9999`;
- validation: `10000–10999`;
- held-out test: `20000–20999`.

Scientifically separate partitions must not overlap. Configuration validation rejects overlapping
partitions and seeds outside the suite's selected partition. A structural reference/held-out pair
uses the same selected episode seeds so that only the environment configuration changes.

Structural comparisons are authoritative in their evaluation YAML: both `reference_environment`
and `held_out_environment` are resolved relative to that YAML file. The CLI rejects a duplicate
`--env-config` for these runs. Both resolved environment contents are fingerprinted, and semantically
identical configurations are rejected rather than being labelled as generalization.

## Metrics

- **Success:** all targets recovered before truncation.
- **Episode length:** joint environment ticks.
- **Coverage:** the union of reachable cells observed by any agent divided by all reachable cells.
- **Duplicated exploration:** redundant per-agent unique observed cells divided by the sum of
  per-agent unique observed cells.
- **Communication cost:** non-silent valid tokens and tokens per active agent-step.
- **Team efficiency:** recovered targets divided by active agent-steps.
- **Return:** shared shaped reward, treated as an optimization diagnostic.
- **Seed generalization gap:** reference success rate minus held-out-seed success rate.
- **Structural generalization gap:** reference-preset success rate minus shifted-preset success rate.

Aggregate summaries report count, mean, standard deviation, median, and deterministic bootstrap
confidence intervals. A random baseline is a validity check, not a performance floor. No result may
be reported without the corresponding run artifacts.

Coverage and duplicated exploration include observations from the final acting transition, even
when termination removes every agent from the Parallel API's live-agent list.

## v0.1 training configuration

Training YAML owns the environment and validation-suite paths, network dimensions, rollout and PPO
schedule, optimizer coefficients, checkpoint/evaluation frequencies, device policy, determinism
mode, and training seed. Paths resolve relative to the owning YAML. Cross-validation requires the
training seed to belong to the declared train partition and periodic evaluation to select the
validation partition. Total joint environment steps and checkpoint/evaluation frequencies align to
complete rollout boundaries; minibatches divide the resulting agent-transition batch exactly.

The smoke configuration is a pipeline test only. It is not evidence of learning quality and cannot
be included in v0.1 scientific results.

## Day 2 rollout reproducibility

One root training seed is expanded with BLAKE2b-based semantic labels; persistent seeds never use
Python's process-randomized `hash()`. The current streams are `network/actor`, `network/critic`,
`policy/sampling`, `environment/<environment-id>/episode/<episode-index>`, and
`evaluation/<evaluation-index>`. An environment's next reset seed therefore depends only on its
stable numeric slot and local episode counter, not worker scheduling or the order in which other
environments finish.

Rollouts are synchronous and environment-major. Actor selection receives only encoded local
observations and the movement/communication masks. Critic evaluation receives only the separately
encoded centralized state. Training-mode action samples consume the explicit policy generator;
evaluation-mode actions use masked argmax and consume no RNG state. Recreating the networks,
collector, and environments from the same root seed reproduces the collected tensors and exact reset
seed history within the recorded software/platform determinism boundary.

The `rollout-smoke` command reports tensor shapes, transition count, completed episode count, root
seed, device, and reset seeds. It performs no optimization, writes no checkpoint, and is not a
benchmark. GAE, PPO losses, gradient updates, and learning claims remain outside Day 2.
