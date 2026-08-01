# Day 6 multi-seed results

## Research question and protocol

Day 6 tests whether the Day 5 negative result was exposure, seed variance, collapse,
exploration, communication, or sparse-success related. The unchanged-reward control used
`configs/training/mappo_day6_longer.yaml`: 16,384 environment transitions, 128 PPO updates,
and the Day 5 optimizer, network, rollout, and evaluation fields unchanged. Root seeds were
0, 1, and 2. Validation seeds were 10000--10019 for every policy; final-test seeds were not
consumed. CPU timings were 338.0 s, 399.5 s, and 513.1 s (seed 0--2).

Rewards remained target recovery +1, completion +5, step cost -0.01, accepted communication
-0.001. Git commit at experiment start: `b7d8758f226fef85d077227bf99ae0cd7d409d9b`.

## Per-seed results

| seed | transitions | agent transitions | updates | episodes | seconds | best success | best targets | best coverage | untrained targets |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16,384 | 49,152 | 128 | 82 | 338.0 | 0.00 | 0.65 | 0.5381 | 0.20 |
| 1 | 16,384 | 49,152 | 128 | 86 | 399.5 | 0.00 | 0.65 | 0.5331 | 0.10 |
| 2 | 16,384 | 49,152 | 128 | 82 | 513.1 | 0.00 | 0.70 | 0.5432 | 0.05 |

Across trained seeds, success was 0.00 ± 0.00 and targets recovered 0.667 ± 0.029;
coverage was 0.5381 ± 0.0051. Exact untrained targets averaged 0.117 ± 0.076.
The trained-minus-untrained target differences were +0.45, +0.55, and +0.65; success
differences were 0.00 for all seeds. Against random (success 0.30, targets 2.80), trained
success differences were -0.30 and target differences were -2.15, -2.15, and -2.10.
Frontier remained 0.90 success and 3.80 targets. Best and latest checkpoints were identical
on validation for all three seeds.

## Behaviour and stability

All stored scalar metrics were finite. No critic divergence, invalid gradient, excessive KL,
or checkpoint inconsistency was observed. The 128 updates and 82--86 completed episodes per
seed materially increased exposure over Day 5, but the final validation curves were flat at
zero success and low recovery. Coverage rose to about 0.54 while task metrics stayed near the
untrained baseline. Movement distributions showed concentration without a useful rescue
strategy; communication remained low and seed-dependent (best-checkpoint mean messages per
episode: 24.0, 24.0, 24.0 in the stored validation summaries, with no rejections).

**Primary classification: exploration-only improvement.** The result is reproducible across
all three seeds: coverage improves, but success, recovery, and efficiency do not approach
random or frontier. This is not evidence of task learning. A secondary observation is weak
behavioral concentration, but numerical stability and the absence of broad collapse make
“policy collapse” an inferior aggregate label.

The entropy trigger was **not met**: although entropy/action concentration was observed, the
stored longer-run diagnostics did not establish the required reproducible stagnant-near-limit
criterion for an entropy-only intervention. No entropy configuration or run was created.

## Candidate and limitations

`configs/training/mappo_candidate.yaml` is an exact copy of the longer-exposure control and is
frozen by `configs/training/mappo_candidate.freeze.json`. Selection used validation success as
the primary metric with predeclared task/behavioral tie-breakers; no test evidence was used.
Three seeds are not statistically definitive, and the validation suite is not a held-out test.
The reward remains sparse and successful experience remains rare. Day 7 is limited to deeper
validation of this frozen candidate and deciding whether one controlled reward experiment is
justified.
