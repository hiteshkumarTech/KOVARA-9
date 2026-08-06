# Roadmap

1. **Phase 0 — implemented:** reliable procedural simulator, baseline policies, metrics, tests,
   artifacts, and engineering foundation.
2. **Phase 1 — implemented for v0.1:** one MAPPO-style algorithm and a reproducible single-machine
   training pipeline. The learned candidate did not achieve full task completion.
3. **Phase 2 — completed once for v0.1:** preregistered held-out seed and structural evaluation,
   classified as exploration transfer without task completion.
   The v0.2 open-source-demo engineering track adds a packaged cross-platform baseline walkthrough
   and public-workflow hardening without changing this evidence or starting a new learning study.
4. **Phase 3:** communication ablations, dynamic roles, teammate failure, and imitation learning.
5. **Phase 4:** broader environments and advanced multi-agent research.
6. **Phase 5:** optional 3D visualization adapter demonstrating already validated behavior.

Each phase requires measured evidence before the next adds complexity.

The Day 10 repository audit and release preparation are complete when
[`day10-final-audit.md`](day10-final-audit.md) is marked passed. Owner-managed PR integration,
repository rename, tag, and release steps remain in the
[`release-checklist.md`](release-checklist.md). New learning research begins after v0.1 and must not
reuse the consumed final-test partition.
