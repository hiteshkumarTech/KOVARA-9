# Day 7 reward experiment results

## Question and diagnosis

Day 7 asked why agents explore and recover some targets but never complete an episode. Day 6
showed 0% success with 0.667 ± 0.029 targets recovered and 0.5381 ± 0.0051 coverage. The
measured bottleneck is **completion**: partial recovery exists, but the final targets are not
recovered before the 200-step truncation. No final-test seed was used.

## Intervention

The predeclared hypothesis selected exactly one change: target-recovery reward 1.0 → 2.0.
Completion, step, communication rewards and every PPO field were unchanged. Discovery bonus,
known-target potential shaping, and horizon increase were rejected because evidence did not
establish discovery, exploitation, or late-horizon progress as the primary bottleneck.

## Shaped results

| seed | transitions | updates | episodes | seconds | success | targets recovered | coverage | efficiency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16,384 | 128 | 82 | 79.5 | 0.00 | 0.40 | 0.5195 | 0.000667 |
| 1 | 16,384 | 128 | 82 | 284.1 | 0.00 | 0.40 | 0.5195 | 0.000667 |
| 2 | 16,384 | 128 | 82 | 180.3 | 0.00 | 0.40 | 0.5195 | 0.000667 |

Shaped success was 0.00 ± 0.00 and recovery 0.400 ± 0.000. Relative to unchanged Day 6,
success changed 0.00 on every seed while recovery decreased by approximately 0.25, 0.25,
and 0.30. Coverage and efficiency also regressed. The shaped intervention is rejected.

## Audit and decision

The environment awards recovery only for newly recovered targets, so repeated identical
behavior cannot farm it. No target locations enter actor observations; reward uses simulator
state only. No oscillation, prolongation, communication farming, or shaped-return/task-metric
divergence was observed. Stored metrics were finite and all three seeds completed.

The final pre-test candidate is the unchanged Day 6 candidate, frozen as
`configs/training/mappo_final_candidate.yaml`. Selection used validation task metrics only.
The shaped run is retained as negative evidence. Three seeds are not definitive, validation
is not test, and sparse rewards remain a risk. Day 8 must perform the first held-out evaluation
of this frozen candidate.
