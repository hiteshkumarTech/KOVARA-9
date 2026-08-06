# Roadmap

1. **Phase 0 — implemented:** reliable procedural simulator, baseline policies, metrics, tests,
   artifacts, and engineering foundation.
2. **Phase 1 — implemented for v0.1:** one MAPPO-style algorithm and a reproducible single-machine
   training pipeline. The learned candidate did not achieve full task completion.
3. **Phase 2 — completed once for v0.1:** preregistered held-out seed and structural evaluation,
   classified as exploration transfer without task completion.
4. **Open-source workflow — implemented under Unreleased:** packaged cross-platform baseline demo,
   visual documentation, public onboarding, and repository-quality hardening. This track adds no
   learning study or scientific claim.
5. **Future research — decision required:** communication ablations, dynamic roles, teammate failure,
   imitation learning, broader environments, or another algorithm each require approved scope and a
   fresh preregistered evaluation plan.

Each research phase requires measured evidence before the next adds complexity. New work must not
reuse the consumed v0.1.0 final-test partition.

The v0.1.0 result, audit, repository rename, tag, and GitHub release are complete. The v0.2
open-source workflow and GitHub presentation work remain under `Unreleased` until reviewed. See the
[release checklist](release-checklist.md) for the historical v0.1.0 process and the
[decision log](decisions.md) for approved trade-offs.
