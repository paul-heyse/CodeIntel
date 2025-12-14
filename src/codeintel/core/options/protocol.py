"""Options protocol definition.

This module defines the protocol that all options/config classes
should implement for consistent behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Self, runtime_checkable


@dataclass(frozen=True)
class ValidationResult:
    """Result from validating options.

    Attributes
    ----------
    ok
        Whether validation passed.
    errors
        List of error messages if validation failed.
    warnings
        List of warning messages.
    """

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def success(cls) -> ValidationResult:
        """Create a successful validation result.

        Returns
        -------
        ValidationResult
            Result indicating validation passed.
        """
        return cls(ok=True)

    @classmethod
    def failure(cls, *errors: str) -> ValidationResult:
        """Create a failed validation result.

        Parameters
        ----------
        *errors
            Error messages describing validation failures.

        Returns
        -------
        ValidationResult
            Result indicating validation failed.
        """
        return cls(ok=False, errors=list(errors))

    @classmethod
    def with_warnings(cls, *warnings: str) -> ValidationResult:
        """Create a successful result with warnings.

        Parameters
        ----------
        *warnings
            Warning messages.

        Returns
        -------
        ValidationResult
            Result with warnings but no errors.
        """
        return cls(ok=True, warnings=list(warnings))

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge with another validation result.

        Parameters
        ----------
        other
            Result to merge with.

        Returns
        -------
        ValidationResult
            Combined result.
        """
        return ValidationResult(
            ok=self.ok and other.ok,
            errors=[*self.errors, *other.errors],
            warnings=[*self.warnings, *other.warnings],
        )


@runtime_checkable
class OptionsProtocol(Protocol):
    """Protocol for all options/config classes.

    Implementations should be frozen dataclasses that support
    validation, default merging, and serialization.

    Examples
    --------
    >>> class MyOptions(OptionsProtocol):
    ...     def validate(self) -> ValidationResult:
    ...         if self.timeout_ms <= 0:
    ...             return ValidationResult.failure("timeout_ms must be positive")
    ...         return ValidationResult.success()
    """

    def validate(self) -> ValidationResult:
        """Validate options and return any issues.

        Returns
        -------
        ValidationResult
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
    "ValidationResult",
]
