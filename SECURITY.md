# Security policy

## Supported versions

Security fixes target the latest code on `main`. The published `0.1.0` release is a research release,
not a production-support commitment. KOVARA-9 supports Python `>=3.12,<3.13`.

## Report a vulnerability

Do not open a public issue for a suspected vulnerability and do not include credentials, private
datasets, exploit details, or sensitive local paths in public logs.

1. Open the repository's **Security** tab on GitHub.
2. Select **Report a vulnerability** if private vulnerability reporting is available.
3. Include the affected version or commit, environment, minimal reproduction, impact, and suggested
   mitigation. Remove unrelated secrets and personal data.
4. If private reporting is unavailable, contact the repository owner through a private contact method
   listed on the GitHub profile and reference KOVARA-9 without disclosing exploit details publicly.

The maintainer will acknowledge the report when capacity permits, validate impact, coordinate a fix,
and credit the reporter if requested and safe. There is no guaranteed response time or bug-bounty
program.

## Project security posture

KOVARA-9 must not read credentials from source files or emit environment variables in logs. Generated
artifacts contain only explicitly selected runtime provenance. Dependencies are locked and should be
upgraded through reviewed lockfile changes. The repository is a research platform and has not been
audited for production or adversarial deployment.
