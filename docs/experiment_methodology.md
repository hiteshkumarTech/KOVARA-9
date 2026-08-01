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
complete rollout boundaries. PPO consumes every valid agent transition and permits a smaller final
minibatch when the valid count is not divisible by `minibatch_size`.

The smoke configuration is a pipeline test only. It is not evidence of learning quality and cannot
be included in v0.1 scientific results.

## Day 2 rollout reproducibility

One root training seed is expanded with BLAKE2b-based semantic labels; persistent seeds never use
Python's process-randomized `hash()`. The current streams are `network/actor`, `network/critic`,
`policy/sampling`, `optimizer/shuffle`,
`environment/<environment-id>/episode/<episode-index>`, and `evaluation/<evaluation-index>`. An
environment's next reset seed therefore depends only on its stable numeric slot and local episode
counter, not worker scheduling or the order in which other environments finish.

Rollouts are synchronous and environment-major. Actor selection receives only encoded local
observations and the movement/communication masks. Critic evaluation receives only the separately
encoded centralized state. Training-mode action samples consume the explicit policy generator;
evaluation-mode actions use masked argmax and consume no RNG state. Recreating the networks,
collector, and environments from the same root seed reproduces the collected tensors and exact reset
seed history within the recorded software/platform determinism boundary.

The `rollout-smoke` command reports tensor shapes, transition count, completed episode count, root
seed, device, and reset seeds. It performs no optimization, writes no checkpoint, and is not a
benchmark.

## Day 3 advantage and optimization semantics

Rollout rewards, team values, terminal bootstrap values, and episode boundaries have shape
`[T, E]`; GAE expands them over the stable active-agent axis to `[T, E, A]`. For active sample
`(t, e, a)`, the temporal-difference residual is

```text
delta[t,e,a] = reward[t,e] + gamma * bootstrap[t,e] - value[t,e]
```

`bootstrap` is zero for a true termination and otherwise is the terminal next-state value captured
before any reset. A truncation can therefore bootstrap its residual, but termination and truncation
both stop the recursive advantage from crossing the episode boundary. The recurrence also stops at
an explicit next-step episode start or inactive agent slot. Invalid slots remain zero. Value targets
are `value + unnormalized_advantage`; optional advantage normalization uses only valid samples,
population variance, and the configured epsilon, and never changes the value targets.

The actor loss uses the joint factored log probability
`log pi(move,message) = log pi(move) + log pi(message)`. With
`ratio = exp(new_joint_log_probability - old_joint_log_probability)`, policy loss is the negative
mean of the minimum of the ordinary and clipped advantage-weighted surrogates. The critic receives
only centralized state features and uses `0.5 * mean((predicted_value - value_target)^2)`. The total
loss is

```text
policy_loss + value_coefficient * value_loss - entropy_coefficient * joint_entropy
```

Only valid active-agent transitions enter any reduction. Diagnostics include factor and joint
entropy, approximate KL `mean((ratio - 1) - log(ratio))`, clip fraction, mean ratio, valid count, and
explained variance when target variance is meaningful.

One Adam optimizer owns the disjoint actor and critic parameter sets. Every PPO epoch shuffles valid
samples with the explicit `optimizer/shuffle` Torch generator, consumes all minibatches including an
uneven last batch, clears gradients, backpropagates, validates finite gradients, clips the combined
gradient norm, steps Adam, and validates all resulting parameters. Invalid shapes or masks, empty
valid batches, non-finite inputs/log probabilities/advantages/returns/ratios/losses/gradients, and
corrupted copied configuration fail with a typed training error rather than being skipped or
clamped.

`update-smoke` collects one configured rollout, computes GAE, performs the configured bounded PPO
epochs, and requires both actor and critic parameters to change while remaining finite. Its output
explicitly identifies an optimization smoke test, not a benchmark or evidence that useful behavior
has been learned.

## Day 4 checkpoint, resume, and evaluation semantics

The full trainer advances only by `rollout_length * num_environments` joint environment steps. Each
iteration collects one rollout, computes GAE, completes all configured PPO epochs, updates progress,
optionally runs deterministic validation, writes the complete metric history atomically, and then
publishes any scheduled checkpoint. A deliberately bounded `--stop-after-environment-steps` value
must use the same rollout alignment and is labelled `bounded`, never complete.

A checkpoint records actor and critic tensors, Adam state, environment-step/update/episode counters,
all prior update records, both explicit Torch generator states, and each collector environment at the
current transition boundary. Resume restores rather than reseeds or replays this state. It rejects
changed training parameters, environment or validation contents, incompatible actor/critic/action
signatures, misaligned counters, malformed metric history, and non-finite model or optimizer state.
Deterministic equivalence is tested by comparing an uninterrupted run with a separately interrupted
and resumed run, including model tensors, optimizer slots, collector state, RNG states, counters, and
metrics.

Scheduled validation and `evaluate-checkpoint` use masked deterministic actor inference from local
observations only. The critic is not constructed for saved-checkpoint evaluation. The
`compare-policies` workflow pairs random, frontier, untrained shared-actor, and checkpoint policies
on the exact configured episode seeds; each policy receives its own complete artifact directory.
Smoke training and validation remain pipeline evidence only.
