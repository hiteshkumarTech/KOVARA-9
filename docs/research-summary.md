# KOVARA-9 v0.1 research summary

KOVARA-9 is a reproducible multi-agent reinforcement-learning research platform that demonstrated
partial exploration transfer but failed to learn full cooperative task completion under the tested
configuration.

## Problem

KOVARA-9 asks whether independently executing agents can learn cooperative behavior that remains
useful on unseen procedural environments. The challenge is not simply to move through a grid: agents
have partial observations, limited communication, shared task rewards, and no actor access to the
global simulator state. Full success requires the team to find and recover every target before a
fixed episode horizon.

The project is deliberately scoped as a reproducible research platform. It is not a physical-robot
stack or operational rescue system.

## Hypothesis

The v0.1 hypothesis was that a parameter-shared decentralized actor, trained with a centralized
critic and MAPPO-style PPO, could improve cooperative task behavior over its exact initialization
and retain that improvement on unseen seeds and a larger held-out structure.

Success rate was the primary metric. Coverage was a predeclared sparse-task diagnostic, not a
substitute for task completion. Random behavior and a handcrafted frontier heuristic provided
non-neural reference points.

## Environment

The simulator is a deterministic PettingZoo Parallel environment with procedurally generated
obstacles, multiple homogeneous agents, target cells, fixed action spaces, and local grid
observations. Agents select a movement action and a limited-budget communication token. Invalid
movement and message actions are masked. Simulator transitions are independent from renderers,
metrics, and training code.

The final evaluation used 100 unseen procedural seeds on both a 12×12, three-agent, four-target
reference environment and a 16×16, four-agent, six-target held-out structure. Communication remained
enabled under the frozen protocol.

## Algorithm

One feed-forward PyTorch actor is shared by every agent. It receives only local observations and
personal action masks. A separate feed-forward critic receives centralized state during training.
Synchronous rollouts feed boundary-aware generalized advantage estimation and clipped PPO updates.
Movement and message distributions remain factored, and evaluation uses deterministic masked argmax.

All stochastic behavior is routed through explicit semantic seed streams. Atomic checkpoints store
the actor, critic, Adam state, collector state, counters, histories, and RNG states. Exact resume is
validated at complete rollout boundaries.

## Engineering contributions

- A typed procedural multi-agent simulator with separate transition, rendering, policy, and metric
  boundaries.
- Enforced CTDE information flow: local actor input and centralized critic input cannot be crossed.
- Deterministic semantic seeds for generation, action sampling, initialization, and optimizer
  shuffling.
- Factored masked actions, boundary-correct GAE, finite-value validation, and complete PPO minibatch
  consumption.
- Atomic checkpoints with exact interruption/resume equivalence.
- Aligned baseline, initialization, trained-policy, and structural comparisons.
- Canonical configuration/model fingerprints, validation-only freeze selection, preregistration,
  one-time test consumption, and immutable result provenance.
- Tests covering public workflows and scientific-integrity failure modes.

## Experimental protocol

Day 5 ran one short training seed. Coverage improved, but success and target recovery did not. Day 6
held the reward and optimizer fixed, trained seeds 0, 1, and 2 for 16,384 environment transitions
each, and again found zero validation success with a reproducible coverage increase. Day 7 changed
only target-recovery reward from 1.0 to 2.0; partial metrics regressed, so the intervention was
rejected. The unchanged Day 6 candidate was frozen using validation evidence only.

Day 8 preregistered one final evaluation before opening the test partition. Random, frontier, three
exact zero-step neural initializations, and three validation-selected checkpoints received the same
ordered seeds and both environments. Model, optimizer, checkpoint, and RNG identities were checked
before and after. The test partition is now consumed and locked.

## Final result

<!-- day8-metric:trained_success_rate=0.0 -->
<!-- day8-metric:random_success_rate=0.14 -->
<!-- day8-metric:frontier_success_rate=0.855 -->
<!-- day8-metric:trained_targets_recovered=0.6133333333333334 -->
<!-- day8-metric:trained_completion_progress=0.12694444444444444 -->
<!-- day8-metric:trained_exploration_coverage=0.5016276777306968 -->

The final classification is **exploration transfer without task completion**.

| Policy group | Successes / episodes | Success | Targets recovered | Completion | Coverage |
|---|---:|---:|---:|---:|---:|
| Random | 28 / 200 | 0.140 | 3.180 | 0.6383 | 0.9043 |
| Frontier heuristic | 171 / 200 | 0.855 | 4.720 | 0.9479 | 0.9036 |
| Exact untrained, three-seed mean | 0 / 600 | 0.000 | 0.1283 | 0.0267 | 0.3991 |
| Frozen trained, three-seed mean | 0 / 600 | 0.000 | 0.6133 | 0.1269 | 0.5016 |

Every trained seed improved targets recovered, completion progress, efficiency, and coverage over
its exact initialization. None improved success. Random outperformed every trained policy on central
task metrics, while the handcrafted frontier heuristic was dramatically stronger.

## Negative finding

Complete cooperative behavior did not emerge under the tested configuration. All trained episodes
reached their horizon without recovering every target. The actors showed more exploration and
partial progress, but discovery-to-recovery conversion remained weak and blocked movement remained
high. A learned representation or changed action preference is not equivalent to learned task
completion.

## Significance

The strongest supported contribution is the research platform and protocol: the experiment is
reproducible, the exact initialization control reveals genuine partial behavioral change, the final
test was preregistered and consumed once, and the negative result remains visible. The result also
shows why strong non-learning baselines matter: handcrafted structure remained substantially more
effective than the trained policy.

The evidence does not establish statistical significance, broad generalization, autonomous rescue,
or production readiness.

## Limitations

Only three training seeds, one final seed partition, and two environment structures were evaluated.
The actor is feed-forward and shares a sparse team reward. Training exposure was limited, evaluation
used deterministic argmax, and baseline stochasticity beyond the explicit episode seeds was not
independently replicated. The simulator abstracts away sensing, dynamics, uncertainty, people, and
real-world safety.

## Next research steps

Future controlled investigations could isolate better credit assignment, staged curricula,
recurrent memory, or explicit coordination mechanisms. Each would require a new preregistered
protocol, validation-only decisions, new untouched test data, and comparison with the frozen v0.1
control. These techniques are hypotheses, not guaranteed solutions.

For full evidence, see the [Day 8 report](day8-final-heldout-results.md),
[methodology](experiment_methodology.md), and [model/system card](../MODEL_CARD.md).
