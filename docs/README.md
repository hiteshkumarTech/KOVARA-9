# KOVARA-9 documentation

This index separates practical guides, system design, and the frozen v0.1.0 research record. Start
with the [root README](../README.md) if you want to install the package and run the demo.

## Run and reuse

- [Open-source demo](open-source-demo.md) — command behavior, artifacts, seed safety, and validation
- [Troubleshooting](troubleshooting.md) — Python, `uv`, shell, CLI, and output-directory fixes
- [Using KOVARA-9 in other projects](using-kovara9-in-other-projects.md) — supported adapter boundary
- [Contributing](../CONTRIBUTING.md) — development setup, quality gates, and review expectations

## Understand the system

- [Architecture](architecture.md) — package boundaries, CTDE information flow, and checkpoints
- [Research methodology](experiment_methodology.md) — experiment design and evaluation controls
- [Reproducibility](reproducibility.md) — seed streams, fingerprints, artifacts, and replay limits
- [Research questions](research_questions.md) — hypotheses and success criteria
- [Decision log](decisions.md) — major engineering and scientific trade-offs

## Read the published result

- [Research summary](research-summary.md) — approachable interpretation of the completed study
- [Final evaluation](day8-final-heldout-results.md) — frozen held-out findings and exact protocol
- [Final evaluation preregistration](day8-final-evaluation-preregistration.md) — decision made before testing
- [Limitations and future work](v0.1-limitations-and-future-work.md) — known constraints and next hypotheses
- [Model card](../MODEL_CARD.md) — intended use, evaluation, limitations, and ethical considerations

The correct high-level finding is **exploration transfer without full task completion**. The learned
actors improved some partial-behavior measures over exact untrained actors, recorded zero full
held-out successes, and remained below the classical baselines. The public demo is not part of that
evaluation.

## Plan and maintain

- [Roadmap](roadmap.md) — approved scope and future decision points
- [Changelog](../CHANGELOG.md) — released and unreleased changes
- [Security policy](../SECURITY.md) — private vulnerability reporting guidance
- [GitHub repository settings](github-repository-settings.md) — manual About, topic, and preview setup
- [Project copy](project-copy.md) — evidence-aligned portfolio and recruiter language
- [GitHub polish audit](github-polish-audit.md) — v0.2 presentation audit and evidence map
- [Open-source demo audit](v0.2-open-source-demo-audit.md) — packaged demo implementation record

## Historical experiment record

The Day 5–Day 10 documents and JSON files are preserved as the immutable v0.1.0 experiment trail.
They should be read in sequence and must not be regenerated or rewritten during documentation work.

- [Day 5 first learning results](day5-first-learning-results.md)
- [Day 6 multiseed results](day6-multiseed-results.md)
- [Day 7 intervention hypothesis](day7-intervention-hypothesis.md)
- [Day 7 reward experiment results](day7-reward-experiment-results.md)
- [Day 8 final held-out results](day8-final-heldout-results.md)
- [Day 10 final audit](day10-final-audit.md)
