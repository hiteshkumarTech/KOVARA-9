# Day 10 final audit

## Audit state

**Passed.** Scientific identities, documented metrics, figures, the recruiter demo, package,
quality gates, and release-preparation documents are internally consistent. Repository-owner PR,
rename, merge, tag, and publication actions remain deliberately unexecuted.

This audit reads committed configuration, freeze metadata, reports, documentation, and package
metadata. It does not open raw final-test episodes, train a policy, or invoke final evaluation.

## Entry gate

- Branch: `v0.1-ten-day-sprint`.
- Entry commit and recorded origin ref:
  `7abf1bd289135919145bfa3e74b7ec19d9a6861a`.
- Entry worktree: clean.
- Day 9 commits: six present, ending at `7abf1bd`.
- Tracked checkpoint/model/run/cache/build/temporary artifacts: none.
- Secret-pattern matches: none.
- Candidate verification: passed; final-test status reported consumed.

## Scientific identity

| Item | Verified identity |
|---|---|
| Candidate canonical fingerprint | `f085babe1905a441fed9c1b9c64076f875a476ab753105ca17ce61d9338ada1c` |
| Candidate file SHA-256 | `cc75394a5d7d99f8e65eea70565c803c157d4ed8b8cc64d0864af13e78a6a64e` |
| Freeze-record SHA-256 | `76701d4534151d29f028f57ef866be9a2b9d74b971864d3616c819f407c3cd53` |
| Reward fingerprint | `e3c4990b5a16e12edbb40da1038b5aa146e4d0643fb866fadbe2ba84b2a4c508` |
| Reference environment | `ce3e4200324577a6ffe22a4c6c60a80915ef0917abdee4e4e30799c39668f47b` |
| Structural environment | `a95dfaf420bf2a45bfc761e7d857d3b4093ee8389af1d4bd0de3f0b81a6e0598` |
| Day 8 report SHA-256 | `4d9de7d35a2cbee5660ad2d21982ad8706f3ba52b85263fd9ab988aea615a2fd` |
| Preregistration SHA-256 | `cde300dcf8cc33f559bf5accaf99f541eafdfc57ed7559a421816a1fb3601b9a` |
| Consumption-record SHA-256 | `059507747e91ef198cc34a3377276d657248375e839ee64c4dfb9e538ec8532f` |
| Dependency-lock SHA-256 | `d7a5dd5651a8c9e678feb9b916153c4eb3da36f24cf4caab8396bab734909f7e` |

The freeze record contains the same canonical candidate fingerprint. Current reward and reference
environment identities match the candidate bindings; the structural identity matches the
preregistration and Day 8 report.

## Partitions and lock

- Training seeds: `0, 1, 2`.
- Validation seeds: every integer `10000` through `10019`, ascending.
- Final-test seeds: every integer `20000` through `20099`, ascending.
- Configured train, validation, and test ranges: `[0,10000)`, `[10000,11000)`, and
  `[20000,21000)`.
- Pairwise overlap: none.
- Consumption record: `complete`; consumed exactly once.
- No post-test tuning: confirmed by the scientific records and unchanged candidate identity.

The audit verifies the guard through non-episode tests. It does not attempt to evaluate a final
seed.

## Result consistency

Final classification: **exploration transfer without task completion**.

| Policy group | Successes / episodes | Success | Targets | Completion | Coverage |
|---|---:|---:|---:|---:|---:|
| Random | 28 / 200 | 0.140 | 3.180 | 0.6383 | 0.9043 |
| Handcrafted frontier | 171 / 200 | 0.855 | 4.720 | 0.9479 | 0.9036 |
| Exact untrained mean | 0 / 600 | 0.000 | 0.1283 | 0.0267 | 0.3991 |
| Frozen trained mean | 0 / 600 | 0.000 | 0.6133 | 0.1269 | 0.5016 |

Training improved partial recovery, completion progress, efficiency, and coverage over every exact
initialization, but it did not improve the primary success metric. Random outperformed the trained
policies on full success, and the handcrafted frontier heuristic outperformed them substantially.

The README, model/system card, research summary, figures, PR description, release notes, and
portfolio wording are checked against the Day 8 JSON by focused tests. A mismatch fails the audit;
recorded Day 8 values are not edited to repair presentation text.

## Package and environment

- KOVARA-9 package version: `0.1.0`.
- Scientific protocol label: `v0.1`; this is historical shorthand for the `0.1.0` release line.
- Python: `3.12.13`.
- Platform: Windows 11 `10.0.26200`.
- Historical Day 8 gate: 253 passed, 1 skipped, 90.19% coverage.
- Current Day 10 gate: 281 passed, 1 skipped, 90.19% coverage.

## Quality-gate record

All commands exited zero. Runtimes are wall-clock seconds on the audit host and are diagnostics, not
benchmarks.

| Gate | Exact command or operation | Runtime |
|---|---|---:|
| Format | `uv run ruff format --check .` | 0.485 s |
| Lint | `uv run ruff check .` | 0.376 s |
| Strict typing | `uv run mypy` | 12.871 s |
| Full tests | `uv run pytest --basetemp <fresh-writable-os-temp>` | 89.001 s |
| Build | `uv build --out-dir <fresh-os-temp-build-directory>` | 18.739 s |
| Fresh-wheel install/import | `uv pip install --python <fresh-venv-python> --no-deps <built-wheel>` | 1.839 s |
| Figure reproduction | `uv run python scripts/generate_result_figures.py --output-dir <fresh-os-temp-directory>` | 0.684 s |
| No-checkpoint demo | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_recruiter_demo.ps1` | 203.623 s |
| Documentation links | PowerShell local Markdown-link scan, 27 files | 0.306 s |
| Citation | `uv run python -c <CITATION.cff PyYAML validation>` | 0.755 s |
| Secret scan | `rg` scoped secret-pattern scan | 0.424 s |
| Tracked-artifact scan | `git ls-files` plus forbidden-path pattern | 0.442 s |
| Consumed guard | focused Day 10 guard test, no episode execution | 86.980 s |

Pytest collected 282 tests: 281 passed and 1 was skipped, with 90.19% branch coverage. MyPy checked
51 source files. The demo used seeds 4242 and 4243, supplied no checkpoint, performed no training,
and did not invoke final evaluation.

Build artifacts were written outside the repository. The audited wheel imported as version 0.1.0:

- Wheel SHA-256: `d46feb7d7e32723c25ddaa73dd8eadf41f4c4043c85f04dbc614dfa01162cb9e`.
- Source-distribution SHA-256:
  `127e0ce379873e91247d7d205146fc123757933f9b9ec6712c87f97adf7a7364`.

The deterministic figure run reproduced manifest SHA-256
`271503cabfe04da7c8c36634f1174aed8afe4e2708d13d9d9400752daed8a3fa` and all six recorded
per-figure hashes. The machine-readable audit contains the complete hash mapping and gate record.

## Scientific-integrity conclusion

**Pass.** The candidate, reward, environments, architecture, hyperparameters, selection, and Day 8
result are unchanged. Negative findings remain prominent and match the machine-readable report. No
training or final evaluation was run during Day 10. The final-test partition remains consumed and
guarded. This audit authorizes owner review of the release-preparation branch; it does not itself
authorize or claim a merge, rename, tag, or GitHub release.
