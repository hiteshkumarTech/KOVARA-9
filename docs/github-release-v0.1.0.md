# KOVARA-9 v0.1.0 release notes

KOVARA-9 v0.1.0 is a reproducible multi-agent reinforcement-learning research platform for
cooperative exploration and generalization in unseen procedural grid environments.

## Included

- A deterministic procedural PettingZoo Parallel environment with partial observations, limited
  communication, action masks, random behavior, and a handcrafted frontier heuristic.
- A parameter-shared decentralized feed-forward actor and separate centralized team-value critic.
- Synchronous MAPPO-style clipped PPO with boundary-aware GAE, advantage normalization, Adam, and
  joint gradient clipping.
- Deterministic semantic seed streams, typed rollout collection, finite-value validation, atomic
  checkpoints, and exact rollout-boundary resume.
- Three-seed validation, exact untrained controls, aligned baseline comparison, and one rejected
  controlled reward intervention.
- A validation-only frozen candidate followed by one preregistered held-out evaluation.
- A one-time final-test consumption lock that prevents tuning workflows from reusing the consumed
  partition.
- Deterministic result figures generated exclusively from the committed Day 8 JSON, including
  per-figure SHA-256 provenance.
- A CPU recruiter demo that requires neither CUDA nor a committed checkpoint.
- Architecture, methodology, research summary, model/system card, reproducibility, limitations,
  audit, citation, pull-request, rename, and release documentation.

## Final scientific result

**Exploration transfer without task completion.** Frozen trained policies achieved 0 successes in
600 held-out episodes. Random achieved 28 successes in 200 episodes; the handcrafted frontier
heuristic achieved 171 in 200. Training consistently improved partial target recovery, completion
progress, efficiency, and exploration coverage over exact initialization, but it did not produce
full cooperative task completion.

No post-test tuning or final-test rerun occurred. This release is a research simulator and
reproducibility artifact, not a real-world rescue system.

## Known limitations

- Three training seeds do not support a statistical-significance claim.
- Evaluation used one final 100-seed subset, two structures, and deterministic masked argmax.
- Actors are feed-forward and training uses a sparse shared reward.
- Random and frontier outperformed all learned policies on full success.
- Checkpoints are intentionally not distributed in Git and must be treated as trusted binary input.
- The environment abstracts away physical dynamics, sensing uncertainty, people, and operational
  safety.

## Verification and installation

See the [final audit](day10-final-audit.md), [reproducibility guide](reproducibility.md), and
[release checklist](release-checklist.md). The package version is `0.1.0`; the Git tag must be
`v0.1.0` and must point to the final merged `main` commit.
