# System architecture

## Boundaries

Configuration is validated before constructing a simulator. The procedural generator produces an
immutable initial world state from an explicit RNG. `GridRescueParallelEnv` owns transition rules
and exposes decentralized PettingZoo observations plus a separate centralized `state()` view.

Policies implement an environment-independent protocol. Renderers consume immutable snapshots and
cannot advance the simulator. Evaluation orchestrates policies and environments, metrics consume
recorded events, and reporting persists typed records.

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

## Extension points

- New environments implement the same parallel environment and snapshot boundaries.
- New policies implement `Policy`; they do not import grid-rescue internals.
- Future trainers implement `Trainer` and own algorithm-specific dependencies.
- A future 3D viewer implements `Renderer` or consumes recorded snapshots without modifying
  simulator dynamics.
