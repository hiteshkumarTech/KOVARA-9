# Final portfolio wording

These versions preserve the v0.1 result: **exploration transfer without task completion**.

## Resume bullet â€” maximum two lines

- Built a deterministic procedural PettingZoo MARL platform with a parameter-shared decentralized actor, centralized critic, MAPPO-style PPO/GAE, exact resume, multi-seed validation, and preregistered held-out evaluation.
  Demonstrated partial exploration transfer but 0/600 learned-policy successes; random reached 28/200 and handcrafted frontier 171/200, motivating honest failure analysis.

## LinkedIn or project-description paragraph

I built KOVARA-9, a procedural PettingZoo multi-agent research platform with partial observations,
limited communication, a parameter-sharing decentralized actor, centralized critic, and
MAPPO-style PPO with GAE. The engineering work emphasizes deterministic seed streams, atomic
checkpoint/resume, multi-seed validation, aligned baselines, configuration fingerprints,
preregistration, and one-time test consumption. The final result was deliberately reported as
**exploration transfer without task completion**: training improved partial recovery and coverage
over every exact initialization, but the frozen policies completed 0 of 600 held-out episodes,
versus 28 of 200 for random and 171 of 200 for a handcrafted frontier heuristic. The project is
evidence of MARL implementation, reproducible research engineering, failure analysis, and honest
scientific communicationâ€”not a claim of a successful rescue agent.

## Approximately 60-second interview explanation

â€œKOVARA-9 was a ten-day multi-agent reinforcement-learning research sprint. I built a deterministic
procedural environment using PettingZoo, where agents act from local observations and limited
messages. The learning system uses one parameter-shared decentralized actor, a centralized critic,
and MAPPO-style PPO with GAE. The harder engineering work was making the experiment trustworthy:
semantic seed streams, action masking, finite-value checks, atomic checkpoints with exact resume,
three-seed validation, exact untrained controls, aligned random and handcrafted frontier baselines,
configuration fingerprints, preregistration, and a one-use test lock.

The scientific result was exploration transfer without task completion. All trained seeds improved
partial recovery and coverage over their own initialization, but the policies had zero successes in
600 held-out episodes. Random succeeded 28 times in 200 episodes, and the handcrafted frontier
heuristic succeeded 171 times. So I would not present this as a successful rescue agent. I present it
as evidence that I can implement MARL systems, run reproducible controlled experiments, diagnose
failure, and communicate negative results without overstating them.â€
