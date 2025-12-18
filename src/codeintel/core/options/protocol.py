"""Options protocol definition.

This module defines the protocol that all options/config classes
should implement for consistent behavior.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from codeintel.core.validation.outcome import ValidationOutcome

type ValidationResult = ValidationOutcome
"""Backwards-compatible type alias for ``ValidationOutcome``."""


@runtime_checkable
class OptionsProtocol(Protocol):
    """Protocol for all options/config classes.

    Implementations should be frozen dataclasses that support
    validation, default merging, and serialization.

    Examples
    --------
    >>> class MyOptions(OptionsProtocol):
    ...     def validate(self) -> ValidationOutcome:
    ...         if self.timeout_ms <= 0:
    ...             return ValidationOutcome.failure("timeout_ms must be positive")
    ...         return ValidationOutcome.success()
    """

    def validate(self) -> ValidationOutcome:
        """Validate options and return any issues.

        Returns
        -------
        ValidationOutcome
            Validation result with errors/warnings if any.
        """
        ...

    def with_defaults(self, defaults: Self) -> Self:
        """Merge with default values, preferring self's non-None values.

        Parameters
        ----------
        defaults
            Default options to merge from.

        Returns
        -------
        Self
            New options with defaults filled in.
        """
        ...

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary for logging/debugging.

        Returns
        -------
        dict[str, object]
            Dictionary representation of options.
        """
        ...


__all__ = [
    "OptionsProtocol",
    "ValidationOutcome",
    "ValidationResult",
]
