"""Execution result types.

This module provides ExecutionResult, an extended result type
for operations that produce row counts and artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from codeintel.core.results.base import BaseResult

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ExecutionResult(BaseResult):
    """Result type for operations producing rows and artifacts.

    Extends BaseResult with fields for tracking database writes
    and artifact production.

    Attributes
    ----------
    row_counts
        Mapping of table names to row counts written.
    artifacts
        Mapping of artifact names to artifact data or paths.
    warnings
        Non-fatal warnings from execution.
    meta
        Additional metadata about the execution.

    Examples
    --------
    >>> result = ExecutionResult.succeeded(
    ...     row_counts={"analytics.metrics": 100},
    ...     duration_s=2.5,
    ... )
    >>> result.row_counts
    {'analytics.metrics': 100}
    """

    row_counts: Mapping[str, int] = field(default_factory=dict)
    artifacts: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    meta: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def succeeded(
        cls,
        *,
        row_counts: Mapping[str, int] | None = None,
        artifacts: Mapping[str, object] | None = None,
        warnings: tuple[str, ...] = (),
        meta: Mapping[str, object] | None = None,
        duration_s: float = 0.0,
    ) -> Self:
        """Create a success result.

        Parameters
        ----------
        row_counts
            Optional mapping of table names to row counts written.
        artifacts
            Optional mapping of produced artifacts.
        warnings
            Optional non-fatal warnings.
        meta
            Optional execution metadata.
        duration_s
            Operation duration in seconds.

        Returns
        -------
        Self
            Success result.
        """
        return cls(
            success=True,
            duration_s=duration_s,
            row_counts=dict(row_counts) if row_counts else {},
            artifacts=dict(artifacts) if artifacts else {},
            warnings=warnings,
            meta=dict(meta) if meta else {},
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        warnings: tuple[str, ...] = (),
        duration_s: float = 0.0,
    ) -> Self:
        """Create a failure result.

        Parameters
        ----------
        error
            Error message describing the failure.
        warnings
            Optional non-fatal warnings collected before failure.
        duration_s
            Operation duration in seconds.

        Returns
        -------
        Self
            Failure result.
        """
        return cls(
            success=False,
            error=error,
            duration_s=duration_s,
            warnings=warnings,
        )

    @classmethod
    def skip(cls, reason: str) -> Self:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping.

        Returns
        -------
        Self
            Skipped result.
        """
        return cls(success=True, skipped=True, skip_reason=reason)

    @property
    def total_rows(self) -> int:
        """Return total rows written across all tables.

        Returns
        -------
        int
            Sum of all row counts.
        """
        return sum(self.row_counts.values())

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result = super().to_dict()
        if self.row_counts:
            result["row_counts"] = dict(self.row_counts)
        if self.artifacts:
            result["artifacts"] = dict(self.artifacts)
        if self.warnings:
            result["warnings"] = list(self.warnings)
        if self.meta:
            result["meta"] = dict(self.meta)
        return result


__all__ = [
    "ExecutionResult",
]
