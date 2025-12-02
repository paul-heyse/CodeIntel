"""Base types for ingestion steps.

This module defines common types used by all ingestion steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepResult:
    """Result from executing an ingestion step.

    Attributes
    ----------
    rows_written
        Total number of rows written across all tables.
    table_counts
        Mapping of table names to row counts.
    errors
        List of error messages encountered.
    skipped
        Whether the step was skipped.
    skip_reason
        Reason for skipping if applicable.
    """

    rows_written: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def success(self) -> bool:
        """Return True if no errors occurred.

        Returns
        -------
        bool
            True if no errors.
        """
        return not self.errors and not self.skipped

    @staticmethod
    def ok(
        rows_written: int = 0,
        table_counts: dict[str, int] | None = None,
    ) -> StepResult:
        """Create a successful result.

        Parameters
        ----------
        rows_written
            Total rows written.
        table_counts
            Optional mapping of table names to counts.

        Returns
        -------
        StepResult
            Success result.
        """
        return StepResult(
            rows_written=rows_written,
            table_counts=table_counts or {},
        )

    @staticmethod
    def fail(error: str) -> StepResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message.

        Returns
        -------
        StepResult
            Failure result.
        """
        return StepResult(errors=[error])

    @staticmethod
    def skip(reason: str) -> StepResult:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping.

        Returns
        -------
        StepResult
            Skipped result.
        """
        return StepResult(skipped=True, skip_reason=reason)


__all__ = ["StepResult"]
