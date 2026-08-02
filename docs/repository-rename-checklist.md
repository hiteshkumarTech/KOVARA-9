# Repository rename checklist

Current GitHub repository name: `KOVARA-9-`

Target GitHub repository name: `KOVARA-9`

The rename has **not** been confirmed or performed by this audit. Do not point automation at the
target URL until GitHub Settings shows the new name.

## Owner actions

1. In GitHub, open **Settings â†’ General â†’ Repository name**, enter `KOVARA-9`, and confirm the
   rename.
2. Only after GitHub confirms the rename, update the local remote:

   ```powershell
   git remote set-url origin https://github.com/hiteshkumarTech/KOVARA-9.git
   ```

3. Verify the new remote and repository reachability:

   ```powershell
   git remote -v
   git fetch origin
   git ls-remote origin
   ```

4. Search tracked text for the former repository URL and trailing-hyphen name. The checklist itself
   may be the final intentional historical mention:

   ```powershell
   git grep -n -I "github.com/hiteshkumarTech/KOVARA-9-"
   git grep -n -I "KOVARA-9-"
   ```

5. Replace confirmed old URLs in documentation and metadata. Do not rewrite experiment names,
   commit history, or scientific records merely because they contain historical provenance.
6. Check GitHub Actions on the renamed repository and confirm both Windows and Ubuntu jobs can fetch
   the repository.
7. Check README badges, local documentation links, and external links.
8. Check README clone instructions after a real URL is inserted.
9. Add or update `repository-code`/`url` fields in `CITATION.cff` only after the target URL resolves.
10. Update recruiter, resume, LinkedIn, and portfolio links.
11. Confirm branches, pull requests, releases, and tags retained their history through the rename.

## Completion evidence

- [ ] GitHub Settings displays `KOVARA-9`.
- [ ] `git remote -v` shows the target URL for fetch and push.
- [ ] `git fetch origin` exits zero.
- [ ] `git ls-remote origin` returns refs.
- [ ] No active documentation or metadata link uses the former URL.
- [ ] CI passes after the rename.
- [ ] The release tag and GitHub release remain reachable.
