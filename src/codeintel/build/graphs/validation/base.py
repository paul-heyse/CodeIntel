"""Base class for graph validation checks implementing CheckProtocol.

This module provides the base class that all graph validation checks
should inherit from to implement the CheckProtocol from core/validation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.validation import ValidationSeverity


class GraphCheckBase:
    """Base class for graph validation checks implementing CheckProtocol.

    Subclasses must define class variables for check metadata and
    implement the execute() method.

    Class Attributes
    ----------------
    check_name
        Unique identifier for this check.
    check_description
        Human-readable description of what this check validates.
    default_severity
        Default severity level for findings from this check.

    Examples
    --------
    >>> class MyCheck(GraphCheckBase):
    ...     check_name = "my_check"
    ...     check_description = "Check for my condition"
    ...     default_severity = "warning"
    ...
    ...     def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
    ...         findings = []
    ...         # ... validation logic ...
    ...         return findings
    """

    check_name: ClassVar[str]
    check_description: ClassVar[str]
    default_severity: ClassVar[ValidationSeverity] = "warning"

    @property
    def name(self) -> str:
        """Return unique check identifier.

        Returns
        -------
        str
            Check name used for filtering and reporting.
        """
        return self.check_name

    @property
    def description(self) -> str:
        """Return human-readable description.

        Returns
        -------
        str
            Description of what this check validates.
        """
        return self.check_description

    @property
    def severity(self) -> ValidationSeverity:
        """Return default severity level.

        Returns
        -------
        ValidationSeverity
            Default severity for findings from this check.
        """
        return self.default_severity

    def __call__(self, ctx: GraphValidationContext) -> Sequence[Mapping[str, object]]:
        """Execute the check and return findings.

        Parameters
        ----------
        ctx
            Context object providing data for validation.

        Returns
        -------
        Sequence[Mapping[str, object]]
            Findings from this check.
        """
        return self.execute(ctx)

    @abstractmethod
    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute the validation check.

        Parameters
        ----------
        ctx
            Graph validation context with all required data.

        Returns
        -------
        list[dict[str, object]]
            List of finding dictionaries.
        """
        raise NotImplementedError


__all__ = ["GraphCheckBase"]
