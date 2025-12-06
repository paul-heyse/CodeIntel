"""Base validation options for all validation subsystems.

This module provides the common options structure used by both
graph validation and ingestion validation frameworks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

ValidationSeverity = Literal["info", "warning", "error"]
"""Severity level for validation findings."""


@dataclass(frozen=True)
class BaseValidationOptions:
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


__all__ = [
    "BaseValidationOptions",
    "ValidationSeverity",
]
