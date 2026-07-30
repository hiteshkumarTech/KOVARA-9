"""Domain-specific exceptions with actionable failure messages."""


class KovaraError(Exception):
    """Base class for expected KOVARA-9 errors."""


class ConfigurationError(KovaraError):
    """Raised when a configuration file cannot be loaded or validated."""


class GenerationError(KovaraError):
    """Raised when a valid procedural world cannot be generated."""


class InvalidActionError(KovaraError):
    """Raised when an agent submits an invalid action."""


class ArtifactError(KovaraError):
    """Raised when evaluation artifacts cannot be written safely."""
