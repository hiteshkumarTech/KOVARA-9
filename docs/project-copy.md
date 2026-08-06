# Project page copy

The text below is aligned with the frozen v0.1.0 evidence. Do not append success claims that are not
supported by the final result record.

## GitHub description

Reproducible multi-agent RL research platform for cooperative exploration in procedural environments.

## GitHub About

KOVARA-9 is an open-source research platform for studying whether independently acting agents learn
coordination that transfers to unseen procedural environments. It includes a deterministic
PettingZoo simulator, MAPPO-style training, classical baselines, held-out evaluation, and a safe CLI
walkthrough. The published result is exploration transfer without full task completion.

## Portfolio project card

**KOVARA-9 — Reproducible Multi-Agent RL Research Platform**

Built an end-to-end procedural MARL research system with decentralized policy inputs, centralized
training state, deterministic seeds, structured artifacts, cross-platform CI, and a public CLI demo.
The preregistered evaluation found improved exploration and partial recovery, but no full held-out
task completion by the trained policies.

## Resume project title

KOVARA-9 — Reproducible Multi-Agent Reinforcement Learning Research Platform

## Resume bullets

- Engineered a typed PettingZoo/PyTorch research stack for procedural cooperative exploration,
  including deterministic generation, MAPPO-style CTDE, checkpoint/resume, paired evaluation, and
  cross-platform packaging.
- Ran a preregistered three-seed study and preserved the negative result: learned policies improved
  exploration and partial target recovery over exact untrained actors but achieved zero full
  held-out successes.

## LinkedIn project description

KOVARA-9 investigates coordination transfer in procedural multi-agent environments. I built the
simulator, policy and MAPPO-style training boundaries, reproducibility controls, structured
evaluation, and an installable cross-platform demo. The strongest outcome was partial exploration
transfer—not task mastery—and the repository exposes the complete frozen evidence and limitations.

## Short recruiter explanation

KOVARA-9 is a complete research-engineering project, not just a training notebook. It turns a MARL
question into a tested simulator, deterministic experiment pipeline, honest held-out evaluation, and
an installable demo. The learned policy improved partial behavior but did not solve the task.

## Thirty-second interview explanation

I wanted to test whether agents using only local observations could learn coordination that survives
new procedural layouts. I built a PettingZoo environment, explicit seed partitions, MAPPO-style
centralized training with decentralized execution, reproducible checkpoints, and paired baselines.
The frozen policies explored and recovered more than their untrained versions, but achieved zero
full held-out successes. I kept that negative result visible and turned the platform into a reusable,
tested open-source package.

## Technical explanation

KOVARA-9 separates procedural transition logic, immutable snapshots, renderers, environment-agnostic
policy protocols, rollout storage, MAPPO-style optimization, and evaluation. Actors consume only
local observations; the critic may consume centralized state during training. All stochastic paths
derive from explicit seed streams, and final evidence is bound to configuration fingerprints,
checkpoint hashes, a preregistration, and a consumed-test lock.

## Honest limitation statement

Under the frozen v0.1.0 configuration, trained actors showed exploration transfer and greater partial
target recovery than exact untrained actors, but recorded no full held-out task successes and
remained below random and handcrafted frontier baselines. The public demo uses baseline behavior and
must not be interpreted as trained-policy performance.
