# Changelog

All notable changes to KOVARA-9 are documented here. The project follows semantic versioning once a
release is tagged. The `0.1.0` entry is prepared, but the tag and GitHub release remain owner actions.

## [Unreleased]

### Added

- A fast, cross-platform `kovara9 demo` command that runs from the repository or an installed wheel.
- A wheel-packaged, strictly validated demo definition with explicit environment parameters,
  baseline policies, seed partitions, episode seeds, and frame limits.
- Deterministic resolved-YAML and JSON demo artifacts, with collision-safe output handling.
- Unit and integration coverage for demo reproducibility, partition safety, packaging, CLI behavior,
  and honest non-benchmark classification.

### Changed

- Evaluation can expose defensive post-transition snapshots through an optional observer adapter;
  rendering remains unable to mutate or advance simulator state.
- CI now validates the packaged demo entry point in addition to formatting, linting, typing, tests,
  and builds.
- Project metadata now declares repository, documentation, issue, and discovery links/keywords.

### Scientific status

- This is an engineering and presentation workflow only. It adds no learning algorithm, training
  run, benchmark, candidate selection, or new scientific result.
- The v0.1 frozen results, candidate identities, and consumed final-test record are unchanged.

## [0.1.0] - 2026-08-02

### Added

- Recruiter-focused README and five-minute CPU demo.
- Mermaid architecture, information-boundary, experiment-lifecycle, and reproducibility diagrams.
- Deterministic SVG result figures generated exclusively from the committed Day 8 JSON.
- Research summary, model/system card, reproducibility guide, limitations/future-work analysis, and
  portfolio metadata guidance.
- Citation metadata and presentation-integrity tests.
- Independent scientific audit, PR/release documents, repository-rename checklist, integration
  procedure, release checklist, and final portfolio wording.

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

The `v0.1.0` tag and GitHub release must be created only after the owner completes the release
checklist on the reviewed merged `main` commit.
