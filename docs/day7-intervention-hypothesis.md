# Day 7 intervention hypothesis

## Diagnosis

The Day 6 frozen policy has a completion bottleneck: validation episodes recover about
0.667 of 4 targets on average but have 0% completion, while coverage is about 0.538.
Exact untrained policies recover only 0.117 targets on average. This indicates useful
partial recovery, not a discovery-only failure; remaining targets are not converted into
terminal completion before truncation.

## One predeclared intervention

Increase only `reward.target_recovery` from `1.0` to `2.0`. Completion bonus (+5), step
penalty (-0.01), communication cost (-0.001), all PPO fields, architecture, horizon, and
seed partitions remain unchanged. The hypothesis is that sparse intermediate recovery
credit is too weak relative to 200-step costs and delayed completion credit.

Rejected alternatives: discovery bonus, known-target potential shaping, and horizon increase.
Entropy, learning rate, and architecture changes are outside this controlled experiment.
Risk is easy-target seeking or prolonged episodes to farm recovery. Falsification is no
improvement in completion and recovery on at least two seeds, or shaped-return improvement
without task-metric improvement. Validation success is primary; recovery, efficiency,
coverage, duplication, collisions, and communication are secondary. Actor observations are
unchanged and contain no global diagnostic state.
