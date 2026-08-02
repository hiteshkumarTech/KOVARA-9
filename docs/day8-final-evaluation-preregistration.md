# Day 8 final held-out evaluation preregistration

Status: **preregistered before generation or execution of any final-test world**. The authoritative
scientific protocol is the matching JSON file. Its byte-level SHA-256 will be passed to the final
command and recorded without editing the protocol after evaluation begins.

## Research questions and frozen candidate

This one final evaluation asks whether the frozen Day 6 shared policy outperforms its exact
untrained initialization on unseen procedural seeds and a held-out structure, whether validation
improvements transfer, how large the validation-to-test gap is, and how the trained policies compare
with random and frontier baselines. The frontier policy is a handcrafted heuristic, not a learned
model.

The candidate is `configs/training/mappo_final_candidate.yaml`. Its canonical fingerprint is
`f085babe1905a441fed9c1b9c64076f875a476ab753105ca17ce61d9338ada1c`; reward fingerprint is
`e3c4990b5a16e12edbb40da1038b5aa146e4d0643fb866fadbe2ba84b2a4c508`; environment fingerprint is
`ce3e4200324577a6ffe22a4c6c60a80915ef0917abdee4e4e30799c39668f47b`. Selection used validation
only. Entry inspection found an extra `c` in the committed textual candidate fingerprint; the clean
scientific snapshot commit `9906383d9321cad5e954f7ed6d454395c552de2f` corrects that transcription
without changing candidate bytes or canonical identity. Before this repair, the original checkout
was at remote commit `3d12c2f92735ff8722e7959f93854eb57480f9a7`. GitHub credentials are not
available in the sandbox, so remote synchronization will require the documented manual commands.

## Partitions, environments, and policies

Training seeds are `0, 1, 2`; previously used validation seeds are `10000-10019`. Final test seeds
are exactly the ordered range `20000-20099`, within the declared `20000-20999` test partition. The
three configured partitions do not overlap.

Every policy receives every final seed on both environments, in ascending order:

- reference structure: `grid_rescue_medium.yaml`, 12x12, three agents, four targets, 200 steps;
- held-out structure: `grid_rescue_hard.yaml`, 16x16, four agents, six targets, 300 steps.

Communication remains enabled with the frozen budgets (8 and 12 per agent respectively). There is
no communication-disabled or reduced-budget ablation. Random and frontier are each executed once
per environment. Exact saved zero-step initial checkpoints and frozen best checkpoints are evaluated
for training seeds 0, 1, and 2. Latest checkpoints and rejected Day 7 shaped policies are excluded.
All neural inference is deterministic masked argmax on CPU.

The exact checkpoint paths, file checksums, actor fingerprints, critic fingerprints, selected
training steps, environment/evaluation fingerprints, and full seed list are recorded in the JSON
preregistration.

## Metrics and aggregation

Task success rate is primary. Secondary metrics are targets recovered, team efficiency, episode
length among successful episodes, and completion progress. Diagnostics are coverage, duplicated
exploration, targets observed, discovery-to-recovery conversion, collisions, blocked movements,
communication use/rejections, and inference latency. If there are no successes, successful-episode
length is reported as not applicable. Raw return remains a within-environment diagnostic and is not
used for cross-environment or cross-reward conclusions.

Targets observed means distinct true targets entering at least one local observation radius;
centralized target state is used only for this metric. Collision counts include movement attempts
blocked by another agent through contention, swaps, cycles, or occupied-agent dependencies, while
blocked movement counts all non-STAY attempts the simulator reports blocked.

Per policy and environment, all 100 episode rows are retained. Summaries include mean, sample
standard deviation, minimum, maximum, and the configured deterministic 95% percentile bootstrap
interval with 2,000 resamples. The primary held-out policy value pools the two 100-episode suites
with equal episode weight. Across the three learned-policy seeds, report mean, sample standard
deviation, minimum, maximum, and a descriptive deterministic 95% percentile bootstrap interval with
10,000 resamples. No statistical significance will be inferred from three training seeds.

Paired differences preserve training seed, environment, and ordered episode seed. Random/frontier
episode rows are reused for alignment without rerunning those baselines. Failed and zero-success
seeds cannot be dropped.

The primary generalization gap is:

```text
generalization_gap = validation_metric - pooled_held_out_metric
```

A positive value is degradation. Validation-minus-reference-test,
validation-minus-structural-test, and reference-test-minus-structural-test are also reported.
Validation comes only from the saved Day 6 aligned validation artifacts.

## Classification and integrity lock

Classification follows the ordered rules in the JSON: evaluation failure first; then reproducible
task learning and transfer; negative transfer; exploration transfer without task completion; weak
or inconsistent transfer; otherwise no meaningful transfer. The rules make success primary and do
not allow coverage alone to become a task-completion claim.

Before the first episode, all candidate and checkpoint identities must pass and the evaluator will
atomically claim `configs/evaluation/final_test_consumed.json`. It fingerprints checkpoint files,
actor tensors, critic tensors, and optimizer state before and after; it also checks neural evaluation
does not change Torch RNG state. An existing consumption record blocks any ordinary final run or
tuning use. A post-claim failure stays visible. Any later authorized replication must use a new
untouched partition.

After results are visible there will be no reward, hyperparameter, architecture, candidate, or
checkpoint-selection change; no test-driven retraining or debugging; and no Day 9 work until the
held-out report and integrity gates are complete.
