"""Base execution report for plugin executors.

This module defines the base execution report dataclass that provides
common fields and metrics used by all domain-specific execution reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.core.plugins.result import PluginExecutionRecord

ExecutionStatus = Literal["succeeded", "failed", "partial"]


@dataclass(frozen=True)
class BaseExecutionReport:
    """Common execution report for all domains.

    Provide the base fields and computed properties for execution reports.
    Domain-specific reports extend this with additional fields as needed.

    Attributes
    ----------
    run_id
        Unique identifier for this execution run.
    started_at
        When execution started.
    ended_at
        When execution ended.
    duration_ms
        Total execution duration in milliseconds.
    records
        Per-plugin execution records.
    fatal_error
        Whether run ended due to fatal error.

    Examples
    --------
    >>> from datetime import datetime, UTC
    >>> report = BaseExecutionReport(
    ...     run_id="test-123",
    ...     started_at=datetime.now(UTC),
    ...     ended_at=datetime.now(UTC),
    ...     duration_ms=100.0,
    ...     records=(),
    ... )
    >>> report.status
    'succeeded'
    """

    run_id: str
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    records: tuple[PluginExecutionRecord, ...] = field(default_factory=tuple)
    fatal_error: bool = False

    @property
    def success_count(self) -> int:
        """Count of successfully executed plugins.

        Returns
        -------
        int
            Number of plugins with succeeded status.
        """
        return sum(1 for r in self.records if r.status == "succeeded")

    @property
    def failure_count(self) -> int:
        """Count of failed plugins.

        Returns
        -------
        int
            Number of plugins with failed status.
        """
        return sum(1 for r in self.records if r.status == "failed")

    @property
    def skip_count(self) -> int:
        """Count of skipped plugins.

        Returns
        -------
        int
            Number of plugins with skipped status.
        """
        return sum(1 for r in self.records if r.status == "skipped")

    @property
    def status(self) -> ExecutionStatus:
        """Derive overall execution status from records.

        Returns
        -------
        ExecutionStatus
            Overall status based on plugin execution outcomes.
        """
        if self.fatal_error:
            return "failed"
        if self.failure_count > 0:
            return "failed"
        if self.skip_count > 0 and self.success_count == 0:
            return "partial"
        if self.skip_count > 0:
            return "partial"
        return "succeeded"

    @property
    def duration_s(self) -> float:
        """Duration in seconds.

        Returns
        -------
        float
            Duration converted to seconds.
        """
        return self.duration_ms / 1000


__all__ = [
    "BaseExecutionReport",
    "ExecutionStatus",
]
