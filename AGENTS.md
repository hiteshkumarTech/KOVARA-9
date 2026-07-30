# Permanent instructions for coding agents

These rules apply to every change in this repository.

1. Preserve the research question: independently acting agents must learn coordination and
   be evaluated on unseen procedural environments.
2. Keep simulator state transitions independent from rendering and user interfaces.
3. Keep algorithms and policies independent from concrete environments. Use declared
   protocols and adapters at subsystem boundaries.
4. Decentralized policies may consume only their agent observation. Centralized state is
   reserved for metrics, debugging, and future CTDE trainers.
5. Put research parameters in validated configuration files. Do not add hidden constants.
6. Route all stochastic behavior through explicit, reproducible seed streams. Never use a
   module-global random generator.
7. Use strong type hints. New production code must pass strict MyPy and Ruff checks.
8. Add unit tests for rules and integration tests for public workflows. Procedural
   invariants require property-based tests when practical.
9. Never fake metrics, benchmarks, training, generated data, or completed functionality.
   Surface errors with context; never silently swallow invalid states or actions.
10. Do not add LLMs, 3D engines, combat, dashboards, databases, microservices, cloud
    infrastructure, or multiple learning algorithms without an approved roadmap decision.
11. Never commit credentials, private data, caches, generated runs, or model artifacts.
12. Update architecture/methodology docs when public interfaces, metrics, rewards, or
    reproducibility behavior changes. Record major decisions and trade-offs.
13. Before completion, run formatting, linting, strict typing, tests, package build, and an
    appropriate CLI smoke check. Report failures honestly.
