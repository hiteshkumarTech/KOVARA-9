# Changelog

All notable changes to KOVARA-9 are documented here. The project follows semantic versioning once a
release is tagged; Day 10 owns the final release decision.

## [Unreleased]

### Added

- Recruiter-focused README and five-minute CPU demo.
- Mermaid architecture, information-boundary, experiment-lifecycle, and reproducibility diagrams.
- Deterministic SVG result figures generated exclusively from the committed Day 8 JSON.
- Research summary, model/system card, reproducibility guide, limitations/future-work analysis, and
  portfolio metadata guidance.
- Citation metadata and presentation-integrity tests.

### Changed

- Documentation now leads with the final classification: exploration transfer without task
  completion.
- Phase and roadmap text now distinguishes implemented v0.1 evidence from future research.
- CI lint scope includes repository scripts.

### Scientific status

- The frozen candidate, reward, architecture, hyperparameters, checkpoints, and final result were not
  changed.
- The consumed final-test partition was not rerun.

## v0.1 experiment record

- Deterministic procedural PettingZoo environment and baseline policies.
- Shared decentralized actor and centralized critic with MAPPO-style PPO and GAE.
- Atomic checkpointing and exact rollout-boundary resume.
- Three-seed validation, one rejected controlled reward intervention, validation-only candidate
  freeze, and one preregistered held-out evaluation.
- Final result: partial exploration transfer, zero learned-policy task successes.

The `v0.1.0` tag and release are intentionally deferred to the Day 10 independent audit.
