"""Base validation options for all validation subsystems.

This module provides the common options structure used by both
graph validation and ingestion validation frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from codeintel.core.options import BaseOptions, ValidationOutcome

if TYPE_CHECKING:
    from collections.abc import Mapping

ValidationSeverity = Literal["info", "warning", "error"]
"""Severity level for validation findings."""


@dataclass(frozen=True)
class BaseValidationOptions(BaseOptions):
    """Base options for controlling validation behavior.

    This dataclass provides the common fields shared by all validation
    option types. Domain-specific validation options extend this base
    with additional fields as needed.

    Attributes
    ----------
    severity_overrides
        Mapping of rule names to severity levels. Use "*" as a key
        to override all rules.
    hard_fail
        Whether to raise an exception on error-level findings.
    max_findings_per_rule
        Maximum findings to collect per rule. None for unlimited.

    Examples
    --------
    >>> opts = BaseValidationOptions(hard_fail=True)
    >>> opts.hard_fail
    True
    >>> opts = BaseValidationOptions(severity_overrides={"*": "error"})
    >>> opts.severity_overrides
    {'*': 'error'}
    """

    severity_overrides: Mapping[str, ValidationSeverity] | None = None
    hard_fail: bool = False
    max_findings_per_rule: int | None = None

    def validate(self) -> ValidationOutcome:
        """Validate the options.

        Returns
        -------
        ValidationOutcome
            Validation result with any errors/warnings.
        """
        errors: list[str] = []

        if self.max_findings_per_rule is not None and self.max_findings_per_rule < 0:
            errors.append("max_findings_per_rule must be non-negative")

        if self.severity_overrides is not None:
            valid_severities = {"info", "warning", "error"}
            for rule, severity in self.severity_overrides.items():
                if severity not in valid_severities:
                    errors.append(
                        f"Invalid severity '{severity}' for rule '{rule}'; "
                        f"must be one of {valid_severities}"
                    )

        if errors:
            return ValidationOutcome.failure(*errors)
        return ValidationOutcome.success()

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
        return type(self)(
            severity_overrides=(
                self.severity_overrides
                if self.severity_overrides is not None
                else defaults.severity_overrides
            ),
            hard_fail=self.hard_fail if self.hard_fail else defaults.hard_fail,
            max_findings_per_rule=(
                self.max_findings_per_rule
                if self.max_findings_per_rule is not None
                else defaults.max_findings_per_rule
            ),
        )


__all__ = [
    "BaseValidationOptions",
    "ValidationSeverity",
]
