# Security

Do not report credentials or private datasets in public issues. Contact the maintainers
privately for suspected vulnerabilities.

KOVARA-9 must not read credentials from source files or emit environment variables in logs.
Generated artifacts contain only explicitly selected runtime provenance. Dependencies are
locked and should be upgraded through reviewed lockfile changes.
