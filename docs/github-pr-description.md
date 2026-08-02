# Pull request: KOVARA-9 v0.1.0

## Objective

KOVARA-9 asks whether independently executing agents can learn coordination from local observations
and limited communication, then transfer that behavior to unseen procedural seeds and a larger
structural environment. This pull request integrates the ten-day v0.1 research sprint without
rewriting its negative result.

## Architecture and engineering

- Deterministic procedural PettingZoo Parallel environment with rendering-independent transitions.
- One parameter-shared decentralized actor receiving agent-local observations and action masks.
- A separate centralized critic used only during MAPPO-style PPO training with GAE.
- Explicit semantic seed streams, finite-value checks, typed subsystem boundaries, and aligned
  policy comparisons.
- Atomic checkpoints containing actor, critic, optimizer, collector, progress, history, and RNG
  state, with tested exact resume at rollout boundaries.
- Canonical configuration and checkpoint fingerprints, validation-only candidate selection,
  preregistration, and a one-time final-test consumption lock.

## Experimental record

- **Day 5:** one short training seed improved coverage but not task completion.
- **Day 6:** three independent seeds, each with 16,384 environment transitions; success remained
  zero while partial recovery and coverage improved.
- **Day 7:** doubling target-recovery reward regressed partial metrics, so the intervention was
  rejected and the unchanged Day 6 candidate was retained.
- **Day 8:** one preregistered held-out evaluation compared random, handcrafted frontier, three
  exact initializations, and three frozen trained policies on aligned unseen seeds and two
  structures.

## Honest final result

**Exploration transfer without task completion.** The frozen trained policies completed 0 of 600
held-out episodes. Random completed 28 of 200, and the handcrafted frontier heuristic completed 171
of 200. Every trained seed improved partial target recovery, completion progress, efficiency, and
coverage over its exact initialization, but training did not improve full-task success. Random and
frontier remained stronger on the primary metric.

No post-test tuning, retraining, candidate reselection, reward change, architecture change, or
final-test rerun occurred.

## Presentation and reproducibility

The branch includes a recruiter-oriented README, architecture diagrams, deterministic JSON-derived
SVG figures, a CPU demo that works without a checkpoint, a model/system card, a research summary,
reproducibility instructions, citation metadata, limitations, and the final scientific audit.

## Verification

The final Day 10 audit records formatting, linting, strict typing, package build/install, demo,
figure, link, citation, secret, artifact, and consumed-test guard results. The local suite collected
282 tests: 281 passed, 1 skipped, with 90.19% branch coverage. Exact commands and runtimes are in
[`docs/day10-final-audit.md`](day10-final-audit.md).

## Known limitations

- Zero learned-policy successes under the frozen configuration.
- Only three training seeds, one final seed subset, and two held-out structures.
- Feed-forward actors, sparse shared reward, and limited training exposure.
- Deterministic argmax evaluates one execution rule.
- Frontier is a handcrafted heuristic, not a learned model.
- The simulator provides no evidence of physical rescue or real-world competence.

## Merge recommendation

Merge with a merge commit after the branch is pushed, CI is green, the full diff is reviewed, and
the repository owner confirms integration. Do not squash the scientific history by default.
