# KOVARA-9 v0.1.0 release checklist

> **Historical record:** the repository owner subsequently completed the rename, merge, `v0.1.0`
> tag, and public GitHub release. The unchecked boxes below preserve the pre-release review template;
> they are not a statement that the current release is pending.

## 1. Final branch audit

- [ ] Day 10 audit status is `passed` in both audit files.
- [ ] Candidate, reward, environment, preregistration, report, and consumption fingerprints match.
- [ ] Final-test consumption status is `complete`; guard tests pass.
- [ ] No scientific configuration changed after Day 8.
- [ ] No training or final-evaluation command was executed during Days 9 or 10.
- [ ] Ruff formatting/lint, MyPy, Pytest, coverage, build, wheel install, demo, figure, links,
      citation, secret, and forbidden-artifact checks pass.
- [ ] Day 10 commits are pushed and the branch worktree is clean.
- [ ] CI is green on `v0.1-ten-day-sprint`.

## 2. Pull request and integration

- [ ] Open a pull request from `v0.1-ten-day-sprint` into `main` using
      [`github-pr-description.md`](github-pr-description.md).
- [ ] Review the complete diff, including all negative Day 5 through Day 8 records.
- [ ] Confirm no generated runs, checkpoints, caches, or build outputs are present.
- [ ] Merge with a merge commit so the scientific history remains visible.
- [ ] Do not squash unless the repository owner explicitly chooses to do so.

Preferred GitHub flow: push the sprint branch, open the PR, wait for CI, review, then merge through
GitHub. Manual fallback, only after owner confirmation and green CI:

```powershell
git checkout main
git pull origin main
git merge --no-ff v0.1-ten-day-sprint
git push origin main
```

After merging:

```powershell
git checkout main
git pull --ff-only origin main
git status --short
git log -1 --decorate --oneline
```

- [ ] Confirm the merged commit contains `docs/day10-final-audit.json` and all Day 10 commits.
- [ ] Run lightweight non-scientific checks: candidate verification, demo `-ValidateOnly`, figure
      regeneration, documentation tests, and package import. Do not rerun final evaluation.

## 3. Repository rename

Complete [`repository-rename-checklist.md`](repository-rename-checklist.md). Rename in GitHub before
changing the local remote. Preserve all Git history, branches, and tags.

## 4. Version and tag

- [ ] `pyproject.toml`, `CITATION.cff`, changelog, release notes, and package metadata say `0.1.0`.
- [ ] `HEAD` is the reviewed merge commit on local and remote `main`.
- [ ] The worktree is clean and no later scientific changes exist.
- [ ] The final audit commit is reachable from `main`.
- [ ] Release notes match the exact tagged contents and retain the negative result.

Create and push the annotated tag only on the final merged `main` commit:

```powershell
git checkout main
git pull --ff-only origin main
git tag -a v0.1.0 -m "KOVARA-9 v0.1.0"
git show --no-patch --decorate v0.1.0
git push origin v0.1.0
```

- [ ] Verify `git rev-parse v0.1.0^{commit}` equals `git rev-parse origin/main`.

## 5. GitHub release

- [ ] Create a GitHub release from tag `v0.1.0` using
      [`github-release-v0.1.0.md`](github-release-v0.1.0.md).
- [ ] Do not attach checkpoints, raw runs, caches, or local build directories.
- [ ] Confirm the release page shows the Apache-2.0 license and citation metadata.
- [ ] Confirm the final classification is visible and random/frontier comparisons are not omitted.
- [ ] Publish only after the owner verifies the renamed repository, tag, and release preview.
