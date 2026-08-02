# Day 8 final held-out results

## Outcome

The preregistered final classification is **exploration transfer without task completion**.
Training improved partial target recovery, completion progress, efficiency, and exploration coverage
over every seed's exact neural initialization, but all three trained policies had **0% success** on
both held-out environments. Random succeeded in 14.0% of episodes and the handcrafted frontier
heuristic succeeded in 85.5%. This is evidence of transferable partial behavior, not evidence of
task completion, broad generalization, rescue intelligence, production readiness, or human-level
performance.

This was the first and only final held-out evaluation of the unchanged Day 6 candidate. No reward,
hyperparameter, architecture, candidate, checkpoint selection, or trained parameter was changed
after test results became visible. There was no retraining, test-seed debugging, candidate
reselection, or Day 9 work.

## Preregistered questions

1. Does each trained policy outperform its exact untrained initialization on unseen procedural
   seeds and environments?
2. Does the validation improvement transfer to the held-out distribution?
3. How large is the validation-to-test generalization gap?
4. How do the trained policies compare with random and frontier baselines?
5. Which positive and negative conclusions are supported?

The answer to the first question is yes for partial-recovery and exploration metrics, but no for the
primary success metric. The partial improvement transferred with a modest average degradation from
validation; it remained far below both non-neural baselines on task metrics.

## Freeze and protocol verification

- Branch at entry: `v0.1-ten-day-sprint`.
- Original local and remote head before the entry repair:
  `3d12c2f92735ff8722e7959f93854eb57480f9a7`.
- Clean scientific evaluation snapshot:
  `6b59739d63c6ef78526bd4a8a00601a829d5830e` (clean at test start).
- Candidate fingerprint:
  `f085babe1905a441fed9c1b9c64076f875a476ab753105ca17ce61d9338ada1c`.
- Reward fingerprint:
  `e3c4990b5a16e12edbb40da1038b5aa146e4d0643fb866fadbe2ba84b2a4c508`.
- Reference-environment fingerprint:
  `ce3e4200324577a6ffe22a4c6c60a80915ef0917abdee4e4e30799c39668f47b`.
- Structural-environment fingerprint:
  `a95dfaf420bf2a45bfc761e7d857d3b4093ee8389af1d4bd0de3f0b81a6e0598`.
- Evaluation fingerprint:
  `e5cbbe68f63cac8f7f60f3ea5d5145892bc2cde5aaad3b504132c1132c41f5ef`.
- Preregistration byte SHA-256:
  `cde300dcf8cc33f559bf5accaf99f541eafdfc57ed7559a421816a1fb3601b9a`.
- Consumption-record SHA-256:
  `059507747e91ef198cc34a3377276d657248375e839ee64c4dfb9e538ec8532f`.
- Analysis SHA-256:
  `a606e7e97ca8c85da45786885c55f327c85374f7aa23c3f0f7eb4d7d2d7f4e22`.

Entry inspection found a one-character transcription error in the freeze record's candidate
fingerprint. The canonical fingerprint calculated from the unchanged candidate was 64 characters;
the recorded string contained an extra `c`. It was corrected before preregistration or test access.
The candidate bytes, reward, environment, training settings, checkpoint selection, and canonical
identity did not change. The original checkout also contained the documented Day 7 protocol files
as untracked files, so the entry worktree was not literally clean. They were preserved in clean
snapshot commit `9906383d9321cad5e954f7ed6d454395c552de2f`; no test seed had been consumed.

All selected checkpoint file, actor, critic, and optimizer fingerprints matched preregistration
before evaluation and were identical afterward. Torch RNG state was unchanged by every neural
suite. No optimizer was stepped. All 1,600 episode rows were retained, aligned, finite, and covered
all eight policies, both environments, and all 100 seeds.

## Candidate and checkpoint provenance

| Training seed | Checkpoint | Selected step | File SHA-256 | Actor SHA-256 |
|---:|---|---:|---|---|
| 0 | exact initialization | 0 | `3be3be7f...9470` | `7fed8569...d9931` |
| 1 | exact initialization | 0 | `9a271db3...e7f2` | `2c73ad51...5b98` |
| 2 | exact initialization | 0 | `0b541c4b...21f3` | `b87f0c15...07b0` |
| 0 | frozen Day 6 best | 16,384 | `406d9177...38fd` | `41e1a5e3...d817` |
| 1 | frozen Day 6 best | 6,144 | `2cb9c837...e19b` | `573a9f0f...1332` |
| 2 | frozen Day 6 best | 14,336 | `95d3efcd...7eb1` | `7cef7263...cd92` |

The authoritative full actor, critic, optimizer, and file checksums are in the preregistration and
consumption JSON records; abbreviated hashes above are for readability.

## Partitions and environments

- Training seeds: `0, 1, 2`.
- Previously used validation seeds: `10000-10019`.
- Final test seeds: every integer from `20000` through `20099`, ascending.
- Configured train, validation, and test partitions: `[0,10000)`, `[10000,11000)`, and
  `[20000,21000)` respectively. Their intersections are empty.
- Reference suite: 12x12 medium structure, three agents, four targets, 200-step limit,
  communication budget 8 per agent.
- Structural suite: 16x16 hard structure, four agents, six targets, 300-step limit,
  communication budget 12 per agent.

Communication was enabled exactly as frozen. No communication-disabled or reduced-budget ablation
was preregistered or run. Latest checkpoints and rejected Day 7 shaped policies were not evaluated.
Every neural policy used deterministic masked-argmax CPU inference.

## Held-out policy results

Values below pool 100 reference and 100 structural episodes with equal episode weight. Successful
length is calculated only on successful episodes; it is not applicable for all neural policies.

| Policy | Success | Targets | Completion | Efficiency | Episode length | Successful length | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.140 | 3.180 | 0.6383 | 0.0045263 | 238.150 | 151.071 (n=28) | 0.9043 |
| Frontier heuristic | 0.855 | 4.720 | 0.9479 | 0.0499792 | 82.415 | 50.193 (n=171) | 0.9036 |
| Exact untrained seed 0 | 0.000 | 0.035 | 0.0079 | 0.0000500 | 250.000 | N/A | 0.3802 |
| Exact untrained seed 1 | 0.000 | 0.165 | 0.0354 | 0.0002167 | 250.000 | N/A | 0.4168 |
| Exact untrained seed 2 | 0.000 | 0.185 | 0.0367 | 0.0002125 | 250.000 | N/A | 0.4002 |
| Frozen trained seed 0 | 0.000 | 0.790 | 0.1633 | 0.0009750 | 250.000 | N/A | 0.5179 |
| Frozen trained seed 1 | 0.000 | 0.535 | 0.1117 | 0.0006708 | 250.000 | N/A | 0.5024 |
| Frozen trained seed 2 | 0.000 | 0.515 | 0.1058 | 0.0006292 | 250.000 | N/A | 0.4846 |

Reference/structural success rates were respectively 0.18/0.10 for random and 0.92/0.79 for
frontier. All six neural policies were 0.00/0.00. Trained reference/structural targets recovered
were 0.76/0.82, 0.54/0.53, and 0.48/0.55 for seeds 0, 1, and 2. Thus the partial behavior existed
on both structures, while task completion existed on neither.

## Training-seed aggregates

These statistics are across the three trained-policy seed values, not across 600 episodes. The
95% intervals are preregistered descriptive seed bootstraps with 10,000 resamples; they are not
significance tests.

| Metric | Mean | Sample SD | Min | Max | 95% descriptive interval |
|---|---:|---:|---:|---:|---:|
| Success | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| Targets recovered | 0.6133 | 0.1533 | 0.5150 | 0.7900 | [0.5150, 0.7900] |
| Completion progress | 0.1269 | 0.0316 | 0.1058 | 0.1633 | [0.1058, 0.1633] |
| Team efficiency | 0.000758 | 0.000189 | 0.000629 | 0.000975 | [0.000629, 0.000975] |
| Episode length | 250.000 | 0.000 | 250.000 | 250.000 | [250.000, 250.000] |
| Exploration coverage | 0.5016 | 0.0167 | 0.4846 | 0.5179 | [0.4846, 0.5179] |

Exact-untrained aggregate means were 0 success, 0.1283 targets, 0.0267 completion, 0.000160
efficiency, 250.0 episode length, and 0.3991 coverage.

## Paired comparisons

Trained-minus-exact-untrained results preserve training seed, environment, and episode seed:

| Seed | Success | Targets | Completion | Efficiency | Coverage | Target + / tie / - episodes | Coverage + / tie / - episodes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | +0.755 | +0.1554 | +0.000925 | +0.1377 | 109 / 88 / 3 | 186 / 1 / 13 |
| 1 | 0.000 | +0.370 | +0.0763 | +0.000454 | +0.0855 | 65 / 127 / 8 | 154 / 6 / 40 |
| 2 | 0.000 | +0.330 | +0.0692 | +0.000417 | +0.0844 | 69 / 110 / 21 | 156 / 4 / 40 |
| Mean | 0.000 | +0.485 | +0.1003 | +0.000599 | +0.1025 | — | — |

Mean trained-minus-random differences were -0.140 success, -2.5667 targets, -0.5114 completion,
-0.003768 efficiency, +11.85 episode steps, and -0.4026 coverage. Mean
trained-minus-frontier differences were -0.855 success, -4.1067 targets, -0.8210 completion,
-0.049221 efficiency, +167.585 episode steps, and -0.4019 coverage. Random outperformed every
trained seed on success and central task metrics; frontier outperformed them by a much larger
margin. Frontier is a handcrafted heuristic, not a learned model.

## Validation-to-test generalization

The preregistered gap is `validation_metric - pooled_held_out_metric`; a positive value means
held-out degradation.

| Seed | Success gap | Target gap | Completion gap | Efficiency gap | Coverage gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | -0.140 | -0.0008 | +0.000108 | +0.0202 |
| 1 | 0.000 | +0.115 | +0.0508 | +0.000413 | +0.0307 |
| 2 | 0.000 | +0.185 | +0.0692 | +0.000538 | +0.0586 |
| Mean validation minus mean test | 0.000 | +0.0533 | +0.0397 | +0.000353 | +0.0365 |

Seed 0 recovered more targets on test than validation, but seeds 1 and 2 degraded. Average partial
recovery and coverage therefore transferred with degradation. Success did not degrade numerically
because it was already zero on validation; this equality does not support task learning.

## Diagnostics and stability

Pooled trained-seed diagnostics were:

| Seed | Duplication | Targets observed | Conversion | Collisions | Blocked moves | Messages | Reference latency |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1915 | 2.435 | 0.2933 | 52.550 | 622.085 | 36.000 | 3.016 ms |
| 1 | 0.1996 | 2.430 | 0.1998 | 36.235 | 661.260 | 35.980 | 3.063 ms |
| 2 | 0.1924 | 2.350 | 0.1846 | 50.830 | 869.885 | 36.000 | 3.794 ms |

Random observed 4.575 targets and recovered 69.3% of observed targets; frontier observed 4.880 and
recovered 96.2%. The trained policies observed only 2.35-2.44 and converted 18.5%-29.3%, with very
large blocked-movement counts. There were no communication rejections and no non-finite values.
Reference-suite inference latencies ranged from 3.016 to 3.794 ms for trained policies; latency is
descriptive wall-clock data, not a benchmark. The v0.1 artifact writer did not serialize structural
suite latency, which is reported as unavailable rather than reconstructed.

The original final command was terminated by the host's 3,600-second command limit after seven
policies had complete durable artifacts and before `trained_seed_2` produced a usable artifact. A
documented continuation, SHA-256
`f20ceb7c07b5820acbc6f8190d1276e87c646bb996bbc44db83a8e729f469b2f`, refused to rerun any
completed policy and executed only the missing policy once. Total final-evaluation wall time was
4,712.814 seconds. This is a documented technical continuation, not a results-motivated rerun.

## Findings and interpretation

Positive findings:

- Every trained seed improved targets recovered, completion progress, efficiency, and coverage
  over its paired exact initialization.
- Partial improvements occurred in both the procedural-seed test and held-out structural suite.
- The evaluator preserved alignment and all model/checkpoint/optimizer identities.
- The strong frontier result demonstrates that the held-out task instances were operationally
  solvable by the existing action and observation interface.

Negative findings:

- The trained candidate completed 0 of 600 episodes; all three seeds had 0% success.
- Random completed 28 of 200 episodes and directly outperformed trained policies.
- Frontier completed 171 of 200 and strongly outperformed trained policies.
- Average validation partial-metric gains degraded on held-out data.
- All trained episodes reached the environment limit, and blocked movement remained severe.
- The evidence does not establish task learning, broad generalization, statistical significance,
  production readiness, or successful rescue intelligence.

## Threats and limitations

- Three independent training seeds are too few for a supported significance claim; intervals are
  descriptive.
- One 100-seed partition and two structures constrain external validity.
- Random and frontier were each run once per episode seed, so baseline stochastic variation beyond
  the explicit episode seeds was not independently replicated.
- Deterministic masked argmax evaluates one deployment rule and not the full learned action
  distribution.
- Reference and structural suites have different target counts and horizons; normalized completion
  and success are the meaningful cross-structure metrics, not raw return.
- Successful-episode length is undefined for neural policies because none succeeded.
- Structural inference timing was not serialized in v0.1.
- The shell-limit continuation is methodologically visible, although it did not duplicate a usable
  policy result or alter the frozen state.

## Test consumption and no-post-test-tuning declaration

The final partition was atomically claimed at `2026-08-02T06:10:26.238680Z` and marked complete at
`2026-08-02T07:28:59.052245Z`. `configs/evaluation/final_test_consumed.json` records the exact
command, seeds, commit, fingerprints, continuation, and checksums for every final artifact. Generic
evaluation and tuning-facing commands reject this consumed partition. Any future rerun must be
labeled a replication and use a new untouched partition.

No post-test tuning occurred. Day 9 has not begun.

## Reproducibility and quality gates

Environment: Windows 11 `10.0.26200`, Intel64 Family 6 Model 142, CPU-only; Python 3.12.13,
PyTorch 2.13.0+cpu, NumPy 2.5.1, Gymnasium 1.3.0, PettingZoo 1.26.1, KOVARA-9 0.1.0.

- `uv run ruff format --check .`: passed.
- `uv run ruff check .`: passed.
- `uv run mypy`: passed.
- Exact requested external pytest directory: sandbox denied creation before test execution.
- Fresh writable OS-temp run: `253 passed, 1 skipped`, 90.19% coverage, exit 0,
  78.539 seconds.
- The requested `E:\KOVARA-BUILD\...` build directory was denied by the sandbox before build
  execution. A fresh OS-temp output outside the repository succeeded in 21.684 seconds and built
  both sdist and wheel. SHA-256: wheel
  `2ba7ca3cd1e3db5e309e76463faf97399adc52eea1e7575e323fc5a272af10bc`; sdist
  `f1cceedb501a6724e212211dce0f52eed535e435863088509b8ba099cb0265c6`.
- CLI smoke: candidate verification passed and reported the final partition consumed; generic
  evaluation of the consumed test configuration exited 2 before creating output or running an
  episode.

Raw run directories, checkpoints, caches, temporary directories, and build outputs remain
untracked. Git index writes and GitHub credentials are unavailable in the original sandbox
checkout, so scientific snapshot commits and exact manual integration commands are reported at
handoff; no push is claimed.
