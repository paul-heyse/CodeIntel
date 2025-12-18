"""Validation outcome primitives.

This module defines a canonical validation result type for cases where validation is modeled
as a boolean outcome with optional error and warning messages.

The project previously accumulated multiple, slightly different ``ValidationResult`` types
across subsystems. ``ValidationOutcome`` is the shared, dependency-free representation that
other layers can reuse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationOutcome:
    """Boolean validation outcome with errors and warnings.

    Attributes
    ----------
    ok
        Whether validation succeeded.
    errors
        Error messages when validation failed.
    warnings
        Warning messages that do not fail validation.
    """

    ok: bool = True
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def success(cls) -> ValidationOutcome:
        """Return a successful validation outcome.

        Returns
        -------
        ValidationOutcome
            Outcome with ``ok=True``.
        """
        return cls(ok=True)

    @classmethod
    def failure(cls, *errors: str) -> ValidationOutcome:
        """Return a failed validation outcome.

        Parameters
        ----------
        *errors
            Error messages describing the validation failures.

        Returns
        -------
        ValidationOutcome
            Outcome with ``ok=False`` and the provided errors.
        """
        return cls(ok=False, errors=errors)

    @classmethod
    def with_warnings(cls, *warnings: str) -> ValidationOutcome:
        """Return a successful validation outcome with warnings.

        Parameters
        ----------
        *warnings
            Warning messages.

        Returns
        -------
        ValidationOutcome
            Outcome with ``ok=True`` and the provided warnings.
        """
        return cls(ok=True, warnings=warnings)

    def merge(self, other: ValidationOutcome) -> ValidationOutcome:
        """Merge this outcome with another outcome.

        Parameters
        ----------
        other
            Outcome to merge.

        Returns
        -------
        ValidationOutcome
            Combined outcome where ``ok`` is the logical AND of both outcomes and
            messages are concatenated.
        """
        return ValidationOutcome(
            ok=self.ok and other.ok,
            errors=(*self.errors, *other.errors),
            warnings=(*self.warnings, *other.warnings),
        )


__all__ = ["ValidationOutcome"]
