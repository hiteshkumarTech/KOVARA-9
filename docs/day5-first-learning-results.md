# Day 5: first learning results

## Conclusion

**No qualifying improvement.** The 4,096-step trained policy changed behavior and increased mean
validation exploration coverage from 42.84% to 48.73% relative to its exact initialization. That
paired coverage change was positive on 14 of 20 seeds, negative on 3, and tied on 3. However,
success stayed at 0%, mean target recovery stayed at 5%, and team efficiency did not change. The
random baseline achieved 30% success, 70% target recovery, and 93.14% coverage. The trained policy
therefore did not improve over both random and untrained policies on a central behavioral metric.

This is a short-run validation result, not a benchmark, generalization result, or successful-learning
claim. No held-out test seed was used.

## Research question and decision rule

The exact question was: **Does the trained shared policy measurably improve over its untrained
initialization and the random baseline on the validation environment?**

The decision rule was fixed before post-training results were inspected. Success rate was primary.
If success was uniformly sparse, exploration coverage was the predeclared secondary signal. Target
recovery, duplicated exploration, team efficiency, communication, and return were retained as
regression checks. Best-checkpoint selection used success, then coverage, team efficiency, lower
duplication, and shorter episode length, in that order.

## Runtime and provenance

- Base Git commit: `41e470354d54f01e28df36ee9bda034e933e324c`.
- Branch: `v0.1-ten-day-sprint`.
- Artifact dirty-tree flag: `true`. Day 5 changes were present, but the Codex filesystem profile
  denied creation of `.git/index.lock`; this is recorded rather than hidden.
- Platform: Windows 11 (`10.0.26200`), AMD64.
- Processor exposed by the shell: Intel64 Family 6 Model 142 Stepping 10, 8 logical processors.
- Python: CPython 3.12.13.
- PyTorch: 2.13.0.
- Training and inference device: CPU.
- Training wall time: 91.17 seconds. CLI process startup and separate evaluation commands are not
  included in that training manifest duration.

## Reproducible inputs

- Training: `configs/training/mappo_day5_short.yaml`.
- Environment: `configs/environments/grid_rescue_medium.yaml`.
- Validation: `configs/evaluation/training_validation.yaml`.
- Training seed: `0` from the declared training partition.
- Validation seeds: `10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008,
  10009, 10010, 10011, 10012, 10013, 10014, 10015, 10016, 10017, 10018, 10019`.
- Training fingerprint: `8c88b26910c0cc10db30ea4247b1881f233c959de3611253102eb12833fbbe9c`.
- Environment fingerprint: `ce3e4200324577a6ffe22a4c6c60a80915ef0917abdee4e4e30799c39668f47b`.
- Validation fingerprint: `2a936c9e551ee20f60d6b29b9efdae2c82bf8f1873e78d8f0b0e16de08d9eb09`.
- Initial actor fingerprint: `7fed8569fba32cfd9f06425478bf78288dc0c982dc4a2aa7c62ed10c3b1d9931`.

The saved zero-step actor, the pre-training checkpoint actor, and both independently reconstructed
untrained actors had that same fingerprint. The trained actor fingerprint was
`da344fd09a97b882f5112bdd78689726a08ea028c9e7005ebde6fda075905ae6`.

## Reward audit

Every acting agent receives the same team reward:

```text
new_targets * 1.0 - 0.01 - accepted_non_silent_messages * 0.001
+ 5.0 if all targets are recovered
```

| Component | Scale | Occurrence | Audit |
|---|---:|---|---|
| Target recovery | +1.0 | Once for each newly occupied target | Directly aligned; cannot be farmed after recovery |
| Success bonus | +5.0 | Once when all four targets are recovered | Strongly aligned with task completion |
| Step penalty | -0.01 | Every joint tick | Encourages speed, but supplies no directional signal early |
| Message penalty | -0.001 | Each accepted non-silent token | Explicit communication cost; at most -0.024 per medium episode |

The maximum plausible medium-environment shared episode return is 8.99: recover all four targets
and finish on the first tick without messaging. With no recovery, 200 steps and all 24 team message
budget units used produce -2.024. Rewards are sparse: there is no direct coverage reward, so an
unsuccessful policy mostly observes the step cost and can immediately reduce loss by suppressing
messages. This can teach silence before messages acquire useful semantics. The step penalty is
ultimately aligned with fast rescue, but it discourages slow exploration before a capable rescue
policy exists. No reward was changed on Day 5 because one short seed is not enough evidence to
change the environment definition.

## Pre-training validation baselines

All policies used the same ordered validation seeds. Neural policies used deterministic masked
argmax. The frontier policy is a heuristic, not a learned policy. Confidence intervals are the
configured deterministic 95% bootstrap intervals.

| Policy | Success (95% CI) | Length | Targets | Recovery | Coverage (95% CI) | Duplication | Efficiency | Messages | Return | Mean inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 30% (10-50%) | 182.55 | 2.8 | 70% | 93.14% (90.42-95.55%) | 44.06% | 0.005645 | 24.0 | 2.4505 | 0.0107 |
| Frontier | 90% (75-100%) | 56.40 | 3.8 | 95% | 86.31% (79.96-91.99%) | 45.18% | 0.051698 | 11.1 | 7.7249 | 0.0771 |
| Exact untrained neural | 0% (0-0%) | 200.00 | 0.2 | 5% | 42.84% (39.70-45.85%) | 13.59% | 0.000333 | 21.3 | -1.8213 | 2.3964 |

The untrained policy was already materially worse than random. Coverage was declared as the sparse
secondary metric before the real run, but a trained policy still had to beat random without central
behavioral regressions to qualify as improvement.

## CPU smoke

The smoke configuration completed 64 environment transitions, 128 agent transitions, 8 optimizer
updates, and periodic validation in 7.91 seconds. Losses, entropies, KL, gradients, parameters,
checkpoints, and artifacts were finite. Latest and best checkpoint paths were created. The smoke
reported always-silent communication; it was treated as a warning only and was not used as a
benchmark or tuning result.

## Real training configuration and counts

The Day 5 configuration preserved `mappo_small.yaml` except that total environment steps were
reduced from 8,192 to 4,096. It used a 128x128 tanh actor and critic, rollout length 64, two
environments, four PPO epochs, minibatches of 128, learning rate 0.0003, clipping coefficient 0.2,
entropy coefficient 0.01, value coefficient 0.5, gamma 0.99, GAE lambda 0.95, advantage
normalization, and maximum gradient norm 0.5.

- Environment transitions: 4,096.
- Agent transitions: 12,288.
- Optimizer updates: 32.
- Completed training episodes: 20.
- Wall time: 91.17 seconds.
- Periodic validation: steps 2,048 and 4,096.
- Best checkpoint: step 4,096.
- Latest checkpoint: step 4,096.
- Best and latest file SHA-256: `e0119919bf8cd1a152108dadec08cd468f1f20cae344cd2fa0042a2cce195a8c`.

## Learning curves

### Validation behavior

| Environment steps | Success | Coverage | Targets recovered | Recovery rate | Duplication | Return |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0% | 42.84% | 0.2 | 5.0% | 13.59% | -1.8213 |
| 2,048 | 0% | 45.97% | 0.1 | 2.5% | 15.43% | -1.9240 |
| 4,096 | 0% | 48.73% | 0.2 | 5.0% | 17.26% | -1.8240 |

Coverage increased, but target recovery first regressed and then only returned to initialization.
Duplication increased throughout. Selection therefore chose step 4,096 on coverage, not on success
or rescue completion.

### Optimization and stability

| Diagnostic | First | Mean | Minimum | Maximum | Last |
|---|---:|---:|---:|---:|---:|
| Total loss | -0.01313 | 0.01860 | -0.02798 | 0.32533 | -0.00592 |
| Policy loss | -0.01298 | -0.00653 | -0.01391 | -0.00027 | -0.00753 |
| Value loss | 0.03682 | 0.08172 | 0.00339 | 0.68628 | 0.02995 |
| Joint entropy | 1.85579 | 1.57362 | 1.33662 | 1.85579 | 1.33662 |
| Approximate KL | 0.00811 | 0.00395 | 0.00021 | 0.00993 | 0.00634 |
| Clip fraction | 0.10482 | 0.05487 | 0.00000 | 0.24805 | 0.10026 |
| Explained variance | 0.47716 | 0.43521 | -0.68816 | 0.96699 | 0.22761 |
| Pre-clip gradient norm | 0.50198 | 0.71999 | 0.40868 | 1.69799 | 0.57656 |
| Post-clip gradient norm | 0.50000 | 0.49462 | 0.40868 | 0.50000 | 0.50000 |

No NaN, infinity, zero-gradient update, excessive-KL warning, all-sample clip saturation, or complete
movement collapse occurred. Gradient clipping behaved as configured. The critic was noisy but did
not monotonically diverge.

Behavior nevertheless concentrated. Movement frequencies changed from
`[17.71%, 13.80%, 22.40%, 20.83%, 25.26%]` to
`[2.86%, 20.57%, 54.69%, 11.72%, 10.16%]`. Joint entropy fell 28%. Message entropy fell from
0.25534 to 0.03630, communication selection fell from 12.5% to 3.125%, and 11 of 32 updates warned
that sampled communication was entirely silent. Deterministic validation still exhausted all 24
team messages after training, showing that stochastic training frequency and argmax evaluation can
tell different stories near a decision boundary.

## Post-training comparison

Best and latest represent the same checkpoint. Their separately written evaluation episode and
summary files were byte-identical.

| Policy | Success (95% CI) | Length | Targets | Recovery | Coverage (95% CI) | Duplication | Efficiency | Messages | Return | Mean inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 30% (10-50%) | 182.55 | 2.8 | 70% | 93.14% (90.42-95.55%) | 44.06% | 0.005645 | 24.0 | 2.4505 | 0.0060 |
| Frontier | 90% (75-100%) | 56.40 | 3.8 | 95% | 86.31% (79.96-91.99%) | 45.18% | 0.051698 | 11.1 | 7.7249 | 0.0802 |
| Exact untrained neural | 0% (0-0%) | 200.00 | 0.2 | 5% | 42.84% (39.70-45.85%) | 13.59% | 0.000333 | 21.3 | -1.8213 | 2.1877 |
| Trained best/latest | 0% (0-0%) | 200.00 | 0.2 | 5% | 48.73% (44.62-52.97%) | 17.26% | 0.000333 | 24.0 | -1.8240 | 2.0933 |

Inference timing is sequential batch-one wall-clock timing on this laptop and is not a behavioral
metric.

## Paired seed outcomes

`S` is success, `T` is targets recovered, and `C` is coverage. Frontier success is included to make
the heuristic comparison explicit.

| Seed | Random S/T | Frontier S | Untrained S/T/C | Trained S/T/C |
|---:|---:|---:|---:|---:|
| 10000 | 0/3 | 1 | 0/0/35.59% | 0/0/39.83% |
| 10001 | 1/4 | 1 | 0/0/44.07% | 0/0/48.31% |
| 10002 | 1/4 | 1 | 0/0/48.31% | 0/0/50.85% |
| 10003 | 0/2 | 1 | 0/0/43.22% | 0/0/56.78% |
| 10004 | 0/3 | 1 | 0/0/33.90% | 0/0/33.90% |
| 10005 | 0/3 | 0 | 0/0/38.14% | 0/0/38.14% |
| 10006 | 0/1 | 1 | 0/0/53.39% | 0/0/49.15% |
| 10007 | 0/3 | 1 | 0/1/41.53% | 0/1/60.17% |
| 10008 | 1/4 | 1 | 0/0/47.46% | 0/0/64.41% |
| 10009 | 0/1 | 1 | 0/0/36.44% | 0/0/35.59% |
| 10010 | 0/2 | 1 | 0/1/47.46% | 0/1/62.71% |
| 10011 | 0/3 | 0 | 0/0/38.98% | 0/0/42.37% |
| 10012 | 0/2 | 1 | 0/0/45.76% | 0/0/53.39% |
| 10013 | 0/3 | 1 | 0/0/35.59% | 0/0/35.59% |
| 10014 | 0/2 | 1 | 0/0/40.68% | 0/0/39.83% |
| 10015 | 1/4 | 1 | 0/0/55.93% | 0/0/58.47% |
| 10016 | 0/2 | 1 | 0/0/46.61% | 0/0/56.78% |
| 10017 | 0/2 | 1 | 0/1/26.27% | 0/1/42.37% |
| 10018 | 1/4 | 1 | 0/0/45.76% | 0/0/52.54% |
| 10019 | 1/4 | 1 | 0/1/51.69% | 0/1/53.39% |

Trained-minus-untrained mean coverage was +5.89 percentage points, with 14 positive, 3 negative,
and 3 tied seeds. Success, target recovery, and efficiency tied on all 20 seeds. Compared with
random, trained-minus-random success was -30 points, target recovery was -2.6 targets, and coverage
was -44.41 points. Random recovered more targets on every seed.

## Diagnosis, failed result, and corrections

The scientific acceptance result failed: the learned actor did not beat random or improve rescue
behavior over initialization. The most conservative diagnosis is insufficient successful experience
under a sparse team reward. Only 20 training episodes completed, and the policy received many
updates before observing enough target or success events. It learned detectable action preferences
and lower communication entropy, but not a better rescue strategy. There is no evidence here for a
simulator defect, exploding optimization, or checkpoint mismatch.

No corrective run was performed. Changing rewards, entropy, or learning rate after one seed would
mix diagnosis with opportunistic tuning. The original configuration and unsuccessful result are
preserved. No architecture, reward, environment fingerprint, or hyperparameter was changed after
the result.

## Reproducibility checks

- Best and latest checkpoints had identical SHA-256.
- Separately generated best-comparison and latest-evaluation episode files had SHA-256
  `de9f7eae65534068c765cac9d15a7e4f6d6b7aea2755c9a48efdc94e3491fb5a`.
- Their summary files had SHA-256
  `07e4ff0d4e1120ce2fff7b1f636d19b789c61e6c6fea0956782c47720656ca7c`.
- Every comparison artifact lists ordered seeds `10000-10019`.
- The tuning comparison command rejects test-partition configuration unless explicitly authorized
  after configuration freeze.
- No checkpoint or run directory is tracked by Git.

## Engineering changes and tests

Changed production behavior:

- exact zero-step initialization checkpoint workflow;
- named-tensor actor fingerprints;
- aligned per-seed policy-comparison records and compatibility fingerprints;
- target recovery aggregates and separate inference-performance artifacts;
- predeclared best-validation checkpoint publication;
- action-factor frequencies and factual stability warnings;
- test-partition guard for tuning comparisons.

Tests cover deterministic initial actor identity, validation selection ordering, warning triggers,
best/latest publication, transition and runtime accounting, paired seed alignment, initialization
fingerprint matching, inference metadata, and test-partition rejection. The complete suite passed
with 220 tests and one CUDA-only skip; branch-enabled coverage was 90.65%. Ruff formatting, Ruff
lint, strict MyPy, package build, and CLI training/evaluation workflows were run. Generated `runs/`
and `dist/` content remains ignored.

## Commands executed

The substantive workflows were:

```powershell
uv run kovara9 config validate configs/training/mappo_day5_short.yaml
uv run kovara9 train --training-config configs/training/mappo_day5_short.yaml --output runs/day5/initial --initialize-only
uv run kovara9 compare-policies --checkpoint runs/day5/initial/checkpoints/step-000000000000.pt --env-config configs/environments/grid_rescue_medium.yaml --eval-config configs/evaluation/training_validation.yaml --output runs/day5/pretraining --device cpu
uv run kovara9 train --training-config configs/training/mappo_smoke.yaml --output runs/day5/smoke
uv run kovara9 train --training-config configs/training/mappo_day5_short.yaml --output runs/day5/training
uv run kovara9 compare-policies --checkpoint runs/day5/training/checkpoints/best.pt --env-config configs/environments/grid_rescue_medium.yaml --eval-config configs/evaluation/training_validation.yaml --output runs/day5/posttraining --device cpu
uv run kovara9 evaluate-checkpoint --checkpoint runs/day5/training/checkpoints/step-000000004096.pt --env-config configs/environments/grid_rescue_medium.yaml --eval-config configs/evaluation/training_validation.yaml --output runs/day5/latest-evaluation --device cpu
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

An inaccessible host UV cache was redirected to the repository's ignored `.uv-cache`. One focused
pytest invocation used an incorrect filename, and a later partial-suite invocation passed its tests
but failed the repository-wide coverage threshold and host temp cleanup. The complete suite was then
run with a writable temporary directory and passed. Git staging/commit failed because the sandbox
could not write `.git/index.lock`; no commit or push is claimed in this report.

## Limitations and remaining risks

- One training seed cannot estimate training variance.
- Twenty completed training episodes are too few for a robust sparse-reward conclusion.
- Validation was observed twice; no final test seed or structural held-out environment was consumed.
- Lower entropy and directional concentration could become policy collapse in a longer run.
- Communication selection and deterministic argmax communication disagree enough to require closer
  inspection before changing communication rewards.
- Day 5 files still require normal-PowerShell commits and a push because Git metadata was read-only
  inside Codex.

## Exact Day 6 scope

Day 6 should use Day 5 evidence only: run controlled two- or three-seed experiments that isolate one
stability hypothesis at a time, beginning with whether longer exposure under the unchanged reward is
enough before changing reward semantics. Compare variance in success, target recovery, coverage,
entropy, action concentration, and communication. If the entropy decline is reproducible, test one
documented entropy-coefficient adjustment while preserving this configuration as the control. Freeze
the final training configuration only after the controlled comparison. Do not consume final held-out
test seeds until that configuration is frozen.

Day 6 was not started.
