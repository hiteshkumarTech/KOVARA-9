# Experiment methodology

## Reproducibility

Every run records resolved configuration, explicit seeds, package version, Python/platform
information, Git revision when available, policy name, and lockfile hash. Environment randomness
uses only episode-local NumPy generators. Deterministic simulator replay does not imply deterministic
future neural-network training.

Seed partitions are fixed:

- training domain: `0–9999`;
- validation: `10000–10999`;
- held-out test: `20000–20999`.

Compared seed suites must not overlap.

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
