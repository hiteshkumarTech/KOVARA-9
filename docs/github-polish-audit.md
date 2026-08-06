# GitHub experience audit

## Audit context

- **Branch:** `v0.2-github-polish`
- **Starting commit:** `4d983559b00359365dfe7e23ca1ba129dbc73182`
- **Package version:** `0.1.0` (unchanged)
- **Scientific classification:** exploration transfer without task completion
- **Audit date:** 2026-08-06

This audit reviews the visitor experience after the cross-platform demo was merged. The local
checkout matches `origin/main` at the starting commit. The public GitHub page could not be rendered
from the audit sandbox, so live About fields, topics, and social preview state require maintainer
verification. Repository content, remote identity, release records, CLI behavior, package contents,
and links were inspected locally.

## 1. What a first-time visitor currently understands

A careful reader can determine that KOVARA-9 is a reproducible multi-agent reinforcement-learning
research platform with a procedural PettingZoo environment, a parameter-shared MAPPO-style learner,
explicit seed streams, checkpoint/resume support, and held-out evaluation. The README states the
negative result honestly and exposes a safe `kovara9 demo` workflow.

That understanding takes too long. The first screen has no visual identity, no badges, no direct
clone URL, and no compact map from “what is this?” to “run it now.” Architecture, result, demo, and
contributor information exist, but the visual hierarchy does not help a 20–30 second visitor.

## 2. What is confusing

- The README repeats the project description before reaching installation.
- `git clone <repository-url>` is still a placeholder after the repository rename.
- The terms “demo,” “evidence tour,” “evaluation,” and “final result” are accurate but not visually
  separated, making it easy for a quick reader to confuse example behavior with published evidence.
- The README roadmap still describes the rename, tag, and release as pending although `v0.1.0` is
  now public and the remote uses `KOVARA-9`.
- The source tree supports external adapters and policy protocols, but there is no focused guide for
  reuse in another project.
- Windows users do not get a complete editable-install path or practical troubleshooting near the
  quick start.

## 3. What looks visually weak

- No logo, banner, social preview, simulator screenshot, or animation exists.
- Existing Mermaid source and deterministic result SVGs are useful but appear well below the fold.
- The README is text-heavy and uses long paragraphs where a callout, compact table, or diagram would
  communicate faster.
- There is no visual distinction between a real demo episode and a frozen scientific chart.

## 4. What creates credibility

- The final conclusion is placed before the research question and explicitly reports zero trained
  held-out successes.
- Frozen JSON, preregistration, configuration fingerprints, a consumed-test lock, and deterministic
  figure provenance are committed.
- The public demo declares safe training-domain seeds and explicit non-benchmark flags.
- CI covers Ubuntu and Windows; Ruff, strict MyPy, Pytest, coverage, and package build gates exist.
- Apache-2.0 licensing, citation metadata, security guidance, changelog, model card, limitations,
  methodology, architecture decisions, and contribution guidance are present.

## 5. Missing or unverifiable GitHub metadata

- No repository-local instructions exist for the exact description, topics, About toggles, or social
  preview image.
- `CITATION.cff` lacks the confirmed repository URL and named maintainer.
- `pyproject.toml` has repository URLs but no maintainer field.
- Live About settings and topics could not be verified from the sandbox.
- There is no repository-local social preview asset.

## 6. Broken or outdated links and state

- The README clone command contains `<repository-url>` instead of the confirmed remote URL.
- Release and rename checklists still speak in future tense. They are valuable historical documents
  but need a completion note so current visitors are not misled.
- Local Markdown links covered by the existing presentation tests resolve. External documentation
  links are limited and point to uv and official PyTorch documentation.
- The GitHub workflow badge does not yet appear because no badge row exists.

## 7. Weak README sections

- The hero has no brand mark, badges, or single obvious command.
- Installation is below architecture, methodology, and results.
- The demo lacks a real screenshot/GIF next to its explanation.
- External reuse and contributor entry points are missing.
- Documentation is not summarized in a scannable index.
- The release-state language is stale.

## 8. Missing visual assets

- Original logo and banner.
- Deterministic static simulator frame.
- Short deterministic animation from a real frontier demo episode.
- Compact, clearly labelled result comparison derived from frozen JSON.
- GitHub social preview.

All new assets must be generated from repository code/data or explicitly described as original
branding/explanatory artwork. No trained-policy screenshot can be produced because checkpoints are
intentionally not distributed.

## 9. Onboarding friction

- Python `>=3.12,<3.13` is documented but the Python 3.13 failure mode is not explained.
- Windows activation policy, uv environments without pip, missing console entry points, editable
  reinstall, and output-directory collisions are undocumented.
- There is no beginner-oriented docs index.
- There are no issue forms, pull-request template, support guide, or code of conduct.

## 10. Recommended changes

### P0 — first-screen clarity and trust

1. Add an original banner/logo, compact real badges, a concise overview, verified result callout, and
   `kovara9 demo` command at the top of the README.
2. Put working Windows and macOS/Linux setup paths near the top.
3. Add deterministic simulator PNG/GIF assets and a frozen-result comparison image with explicit
   provenance and non-benchmark labels.
4. Separate demo evidence from published held-out evidence visually and in prose.
5. Replace stale clone/release/rename language.

### P1 — documentation and contributor experience

1. Add a docs index, troubleshooting guide, and external-project reuse guide.
2. Add actual-code Mermaid diagrams for training/evaluation, packages, and adapter reuse.
3. Add focused issue forms, pull-request template, support policy, and code of conduct.
4. Add exact manual GitHub settings and polished project copy.
5. Add named maintainer metadata while retaining KOVARA-9 Contributors credit.

### P2 — maintainability and regression protection

1. Test visual generation determinism, dimensions, provenance, and safe seeds.
2. Expand link/YAML/SVG/community-file checks.
3. Document a repeatable social-preview and README-asset generation command.
4. Keep historical release checklists but mark externally completed operations clearly.

## 11. Things that must remain untouched

- `configs/evaluation/final_test_consumed.json` and the consumed test partition.
- Frozen candidate, freeze, reward, environment, and training configurations.
- Day 5–10 scientific JSON/Markdown records.
- Existing deterministic result SVGs and `docs/assets/results/manifest.json`.
- v0.1.0 package version, tag, release evidence, and final classification.
- The local-observation boundary for decentralized policies.
- The prohibition on post-test tuning, final-test reuse, fabricated metrics, and unsupported claims.

## 12. Evidence used for project claims

| Claim | Repository evidence |
|---|---|
| Package version is 0.1.0 | `pyproject.toml`, `CITATION.cff`, `docs/day10-final-audit.json` |
| Supported Python is 3.12 only | `pyproject.toml` declares `>=3.12,<3.13`; CI tests Python 3.12 |
| Final classification | `docs/day8-final-heldout-results.json` → `classification_label` |
| Random: 28/200 successes | Day 8 JSON → random pooled successful count and success rate |
| Frontier: 171/200 successes | Day 8 JSON → frontier pooled successful count and success rate |
| Trained: 0/600 successes | Day 8 JSON training-seed aggregates and Day 10 audit |
| Partial behavior improved over initialization | Day 8 JSON → positive mean paired deltas for targets, completion, efficiency, and coverage |
| Demo seeds are safe examples | `src/kovara9/resources/open_source_demo.yaml` and `DemoConfig` validation |
| Demo performs no training/final evaluation | `src/kovara9/demo.py` report flags and CLI output |
| Demo example outcomes | Executed `kovara9 demo --no-render` on Python 3.12.13 |
| CI and quality gates | `.github/workflows/ci.yml`, `pyproject.toml`, and final verification in this branch |
| Reproducible figures | `scripts/generate_result_figures.py` and `docs/assets/results/manifest.json` |
| Apache-2.0 | `LICENSE`, `CITATION.cff`, `pyproject.toml` |

Quality/test counts in the final handoff will be reported only from commands executed after all
polish changes are complete.

## 13. Implemented audit response

The branch addresses every P0 item and the high-value P1/P2 items without changing production
learning behavior:

- The README now opens with original local branding, four factual badges, the honest result, the
  public command, and tested Windows/macOS/Linux installation.
- A deterministic generator creates one real simulator frame, a 10-frame GIF, a frozen-data-derived
  comparison chart, a social preview, and a provenance/hash manifest.
- Architecture documentation now shows training/evaluation, package boundaries, and external reuse.
- A documentation index, troubleshooting, reuse guidance, settings instructions, project copy,
  community forms, support policy, and code of conduct remove the main onboarding gaps.
- Maintainer and canonical repository metadata now coexist with the existing contributor credit.
- Tests lock asset determinism, evidence values, safe seed use, SVG accessibility, and issue-form YAML.

## 14. Five-persona review

| Persona | What is now answered | Remaining boundary |
|---|---|---|
| Recruiter, 30 seconds | Hero explains the project, runnable command, result, engineering scope, and limitation before the research detail | About text and social preview still require the owner to apply GitHub settings |
| ML engineer | Capability table, package diagram, strict quality commands, source layout, and protocol boundaries show how the system is engineered | The project remains one research stack, not a general production platform |
| Researcher | Preregistration, exact result table, evidence links, seed partitions, consumed-test lock, threats, and citation are explicit | The evidence is limited to the completed v0.1.0 study and must not be generalized beyond it |
| Beginner | Python 3.12 setup appears near the top; Windows and Unix commands, CLI variants, and troubleshooting are concrete | Installing PyTorch still depends on the user's platform and can take time |
| Open-source contributor | Issue forms route bugs, reproducibility, features, and environments; the PR template and contribution guide state the required tests and boundaries | Maintainers must label real scoped issues before claiming any task is a “good first issue” |

For all five personas, the repository now answers what KOVARA-9 is, why it matters, how to run it,
what worked, what failed, how it is structured, how to contribute, and how to cite it. During this
review, two additional accuracy fixes were made: the contributor setup now uses the repository's
actual `uv sync --locked` workflow, and capability text correctly describes the implemented
feed-forward MAPPO-style learner rather than implying a recurrent policy.

## 15. Final verification record

Executed on Windows with Python 3.12.13:

| Gate | Result |
|---|---|
| Ruff format | 90 files formatted; check passed |
| Ruff lint | Passed |
| Strict MyPy | Passed across 53 source files |
| Pytest | 296 passed, 1 CUDA-only skip |
| Coverage | 90.50% branch coverage; 90% threshold met |
| Build | `kovara9-0.1.0.tar.gz` and `kovara9-0.1.0-py3-none-any.whl` built successfully |
| Editable install | Fresh temporary editable install and CLI smoke passed |
| Installed wheel | Imported from a fresh temporary target; help, validation, and no-render demo passed |
| README assets | Byte-determinism test passed; GIF decoded as 480×352 with 10 frames |
| Documentation | 106 local Markdown targets resolved across 41 files |
| Structured files | 25 YAML/CFF documents and 8 SVG/XML documents parsed |
| Security hygiene | No high-confidence secret match; no tracked caches, runs, checkpoints, or builds |
| Frozen integrity | No diff in Day 5–10 scientific records, final-test lock, or frozen result assets |

`uv sync --locked` resolved 73 packages and built the project, but the sandbox denied replacement of
the pre-existing workspace virtual environment's editable `.pth` file. The package was therefore
verified through fresh temporary editable and wheel installs instead; both passed. This is an
environment permission constraint, not a lockfile or package-build failure.
