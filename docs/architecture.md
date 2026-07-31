# System architecture

## Boundaries

Configuration is validated before constructing a simulator. The procedural generator produces an
immutable initial world state from an explicit RNG. `GridRescueParallelEnv` owns transition rules
and exposes decentralized PettingZoo observations plus a separate centralized `state()` view.

Policies implement an environment-independent protocol. Renderers consume immutable snapshots and
cannot advance the simulator. Evaluation orchestrates policies and environments, metrics consume
recorded events, and reporting persists typed records.

Each decentralized observation contains the local grid, teammate token slots, the agent's remaining
communication budget, and fixed-shape `move_action_mask` and `message_action_mask` arrays. Action
spaces never change during an episode. A non-silent token selected after the budget reaches zero is
converted to a rejected no-op: it is not broadcast, charged, or counted, and the agent receives
`communication_rejected: true` in its personal transition info.

```text
config → generator → environment → observations → policies
                         ↓              ↓
                 world snapshots     actions
                         ↓              ↓
                    renderer       joint transition
                                        ↓
                              metrics → artifacts
```

## CTDE boundary

Future trainers may consume centralized state during optimization. Executing policies receive only
their local observation and personal memory. This separation is enforced by distinct interfaces
instead of a runtime flag inside a shared observation object.

The centralized `state()` view contains the full obstacle/target/recovery map, ordered agent-position
channels, active-agent slots, ordered communication budgets, ordered latest-message tokens, and the
step counter. It is never passed to baseline policies. Policy outcome info is restricted to personal
movement blocking, personal accepted-message status, and personal communication rejection. Team
recoveries, accepted-message totals, success, and other evaluator facts are available through the
immutable environment-level `last_events` record and snapshots instead of policy info.

## v0.1 learner boundary

The v0.1 learner uses one `SharedActor` instance for every homogeneous agent and a separate
`CentralizedCritic`. `ActorObservationEncoder` accepts only values contained by the decentralized
observation space and produces an `ActorInput`; `CentralStateEncoder` accepts `state_space` values and
produces a distinct `CriticInput`. Both the type boundary and runtime validation reject crossing
these inputs.

Networks are feed-forward MLP foundations with independently derived initialization seeds. The actor
has separate movement and message logits, while action masking and sampling remain outside the
network so the environment contract stays explicit. Fixed-shape rollout storage records local actor
features, centralized critic features, both masks, actions, log probabilities, shared rewards,
values, live-agent slots, and termination/truncation boundaries. Collection, GAE, and optimization
are subsequent sprint milestones and are not implemented by these foundations.

## Extension points

- New environments implement the same parallel environment and snapshot boundaries.
- New policies implement `Policy`; they do not import grid-rescue internals.
- Future trainers implement `Trainer` and own algorithm-specific dependencies.
- A future 3D viewer implements `Renderer` or consumes recorded snapshots without modifying
  simulator dynamics.
