## Summary

Describe the problem and the smallest implemented change.

## Evidence

- Related issue or decision:
- Commands run:
- Test results:
- Documentation updated:

## Research and architecture checks

- [ ] Simulator transitions remain independent from rendering and UI code.
- [ ] Policies and algorithms remain independent from concrete environments.
- [ ] Decentralized policies consume only their agent observation.
- [ ] New research parameters are validated configuration, not hidden constants.
- [ ] New stochastic behavior uses explicit reproducible seed streams.
- [ ] Scientific claims link to real artifacts; no metrics or functionality are fabricated.
- [ ] Frozen v0.1.0 results and the consumed final-test decision are unchanged.

## Quality checks

- [ ] Ruff formatting and linting pass.
- [ ] Strict MyPy passes.
- [ ] Relevant unit, property, integration, and CLI tests pass.
- [ ] The package builds and the installed CLI smoke check passes when applicable.
- [ ] Markdown links and changed YAML/XML files validate.
- [ ] No credentials, caches, generated runs, checkpoints, or private data are included.

## Reviewer notes

Call out limitations, deferred work, compatibility impact, and any manual verification.
