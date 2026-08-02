# KOVARA-9 v0.1 model and system card

## System purpose

KOVARA-9 v0.1 is a reproducible software research platform for studying cooperative exploration and
task completion by multiple reinforcement-learning agents in procedural grid environments. Its
frozen learned candidate is an experimental shared policy, not a deployable agent.

Final classification: **exploration transfer without task completion**.

## Intended research use

- Study procedural multi-agent environments and partial observability.
- Inspect centralized-training/decentralized-execution boundaries.
- Reproduce PPO, GAE, deterministic checkpoint/resume, and aligned evaluation workflows.
- Compare learned behavior with exact initialization, random action, and handcrafted heuristics.
- Teach or review scientific controls, preregistration, test consumption, and negative-result
  reporting.

## Out-of-scope uses

Do not use KOVARA-9 for physical rescue, emergency response, navigation around people, resource
allocation, safety-critical decisions, surveillance, weapons, autonomous deployment, or any claim
about real-world competence. The software has not been validated on physical sensors, robots,
humans, changing hazards, adversaries, or open-world conditions.

## Environment assumptions

The simulator is a discrete two-dimensional grid. Obstacles are static, targets are stationary,
agents are homogeneous, action and observation spaces are fixed, transitions are synchronized, and
the configured team reward is shared. Agents receive local grid observations, local teammate token
slots, a personal remaining communication budget, and action masks. The critic alone receives the
centralized simulator state during training.

The final reference environment was 12×12 with three agents, four targets, and a 200-step horizon.
The structural environment was 16×16 with four agents, six targets, and a 300-step horizon.

## Actor and critic architecture

- **Actor:** parameter-shared feed-forward MLP, two 128-unit tanh hidden layers, separate movement
  and communication logits, masked factored actions, local input only.
- **Critic:** separate feed-forward MLP, two 128-unit tanh hidden layers, centralized state input,
  scalar shared-team value.
- **Training:** synchronous MAPPO-style clipped PPO with GAE, advantage normalization, one Adam
  optimizer over disjoint actor/critic parameters, and joint gradient clipping.
- **Evaluation:** deterministic masked argmax; the critic is not used for decentralized execution.

This implementation has no recurrence, transformer, planning module, world model, curriculum,
distributed collector, value clipping, or target-KL stopping.

## Training data generation

All experience is generated online inside the procedural simulator. There is no human, web,
personal, proprietary, or third-party training dataset. Training seeds were `0`, `1`, and `2`.
Each final candidate training run used 16,384 joint environment transitions under the unchanged Day
6 protocol. Environment generation, policy sampling, network initialization, and optimizer shuffling
use separate deterministic seed streams derived from the root seed.

## Reward structure

Every acting agent receives the same team reward:

```text
new targets × 1.0 - 0.01 per step - 0.001 per accepted non-silent message
+ 5.0 when all targets are recovered
```

The Day 7 experiment doubled only the target-recovery term and produced worse validation partial
metrics. That shaped candidate was rejected. The final candidate retained the unchanged Day 6
reward above.

## Seed partitions and selection

- Training partition: `[0, 10000)`; controlled root seeds `0, 1, 2`.
- Validation partition: `[10000, 11000)`; candidate-selection seeds `10000–10019`.
- Final-test partition: `[20000, 21000)`; one preregistered 100-seed subset was consumed.

Candidate and checkpoint selection used validation data only. The final-test partition is marked
consumed and cannot be used by ordinary evaluation or tuning workflows.

## Final metrics

<!-- day8-metric:trained_success_rate=0.0 -->
<!-- day8-metric:random_success_rate=0.14 -->
<!-- day8-metric:frontier_success_rate=0.855 -->
<!-- day8-metric:trained_targets_recovered=0.6133333333333334 -->
<!-- day8-metric:trained_completion_progress=0.12694444444444444 -->
<!-- day8-metric:trained_exploration_coverage=0.5016276777306968 -->

| Policy group | Episodes | Success | Targets recovered | Completion progress | Coverage |
|---|---:|---:|---:|---:|---:|
| Random | 200 | 0.140 | 3.180 | 0.6383 | 0.9043 |
| Handcrafted frontier | 200 | 0.855 | 4.720 | 0.9479 | 0.9036 |
| Exact untrained mean | 600 | 0.000 | 0.1283 | 0.0267 | 0.3991 |
| Frozen trained mean | 600 | 0.000 | 0.6133 | 0.1269 | 0.5016 |

Trained-minus-exact-untrained means were `+0.485` targets, `+0.1003` completion, and `+0.1025`
coverage, with no success improvement. The trained policy remained below random and frontier.

No statistical-significance claim is supported by three training seeds.

## Known failure modes

- Zero full-task success for all three trained policies on both held-out structures.
- Incomplete target discovery and weak conversion from discovery to recovery.
- Frequent blocked movement, including agent-agent interference.
- Deterministic policies that continue until the environment horizon.
- Low and seed-dependent learned communication behavior.
- Partial exploration improvement that can look positive while the primary task remains unsolved.
- Potential incompatibility with environment configurations whose actor observation/action signature
  differs from the checkpoint definition.

## Limitations

The simulator is intentionally small and abstract. Three training seeds do not characterize the
full optimization distribution. One final seed subset and two structures constrain external
validity. The reward is sparse, the actor lacks memory, and deterministic argmax reveals only one
execution rule. Wall-clock inference measurements are platform-specific and are not performance
benchmarks. Structural-suite latency was not serialized in v0.1.

## Ethical and safety considerations

Grid targets and “rescue” terminology are task abstractions. Success in this simulator would not
imply competence around people, hazards, uncertain maps, damaged infrastructure, legal constraints,
or safety procedures. The actual v0.1 learned policies did not succeed even in the simulator.
Presentations must keep this distinction visible and must not anthropomorphize partial coverage as
judgment, intent, or intelligence.

## Reproducibility

The repository pins dependencies with `uv.lock`, validates YAML through typed schemas, derives RNG
streams semantically, records canonical fingerprints, and publishes checkpoints only at complete
rollout boundaries. The final evaluation was byte-preregistered and produced an atomic consumption
record. JSON-derived figures can be regenerated without raw evaluation runs.

See [docs/reproducibility.md](docs/reproducibility.md) and the
[final held-out report](docs/day8-final-heldout-results.md).

## Checkpoint trust warning

Model checkpoints are intentionally excluded from Git. PyTorch checkpoint files are complex binary
inputs and should be loaded only from a trusted source after verifying the expected SHA-256. The
committed preregistration records the approved local checkpoint checksums, but possession of a
matching file is not implied. Never accept an arbitrary checkpoint merely to run the demo.

## No-production-use statement

**KOVARA-9 v0.1 and its policies must not be used in production or for real-world rescue.** They are
research artifacts for a procedural simulator, and the frozen policies failed full task completion
in the final held-out evaluation.
