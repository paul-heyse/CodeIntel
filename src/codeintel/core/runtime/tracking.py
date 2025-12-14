"""Execution tracking types.

This module provides types for tracking step execution, timing,
and result aggregation.

Note
----
For the unified result protocol and base types, see `codeintel.core.results`.
The `StepResult` type in this module is specifically designed for step-based
execution tracking and maintains its own status enum for backward compatibility.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping


class StepStatus(StrEnum):
    """Status of a step execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StepResult:
    """Structured result from a step execution.

    Attributes
    ----------
    step_name
        Name identifying the step.
    status
        Execution status.
    duration_s
        Execution duration in seconds.
    message
        Optional status message.
    error
        Optional error information.
    outputs
        Optional output data from the step.

    Examples
    --------
    >>> result = StepResult.success("build", duration_s=1.5)
    >>> result.ok
    True
    >>> result.status
    <StepStatus.SUCCESS: 'success'>
    """

    step_name: str
    status: StepStatus
    duration_s: float = 0.0
    message: str | None = None
    error: str | None = None
    outputs: Mapping[str, object] | None = None

    @property
    def ok(self) -> bool:
        """Return True if step completed successfully.

        Returns
        -------
        bool
            True if status is SUCCESS or SKIPPED.
        """
        return self.status in {StepStatus.SUCCESS, StepStatus.SKIPPED}

    @classmethod
    def success(
        cls,
        step_name: str,
        *,
        duration_s: float = 0.0,
        message: str | None = None,
        outputs: Mapping[str, object] | None = None,
    ) -> StepResult:
        """Create a success result.

        Parameters
        ----------
        step_name
            Name identifying the step.
        duration_s
            Execution duration in seconds.
        message
            Optional status message.
        outputs
            Optional output data.

        Returns
        -------
        StepResult
            Success result.
        """
        return cls(
            step_name=step_name,
            status=StepStatus.SUCCESS,
            duration_s=duration_s,
            message=message,
            outputs=outputs,
        )

    @classmethod
    def failure(
        cls,
        step_name: str,
        error: str,
        *,
        duration_s: float = 0.0,
        message: str | None = None,
    ) -> StepResult:
        """Create a failure result.

        Parameters
        ----------
        step_name
            Name identifying the step.
        error
            Error description.
        duration_s
            Execution duration in seconds.
        message
            Optional additional message.

        Returns
        -------
        StepResult
            Failure result.
        """
        return cls(
            step_name=step_name,
            status=StepStatus.FAILED,
            duration_s=duration_s,
            message=message,
            error=error,
        )

    @classmethod
    def skipped(cls, step_name: str, reason: str | None = None) -> StepResult:
        """Create a skipped result.

        Parameters
        ----------
        step_name
            Name identifying the step.
        reason
            Optional reason for skipping.

        Returns
        -------
        StepResult
            Skipped result.
        """
        return cls(
            step_name=step_name,
            status=StepStatus.SKIPPED,
            message=reason,
        )


@dataclass
class TimingContext:
    """Context for tracking timing of an operation.

    Attributes
    ----------
    name
        Name of the timed operation.
    start_time
        Start time as a float (time.perf_counter).
    end_time
        End time, set when stop() is called.

    Examples
    --------
    >>> ctx = TimingContext("operation")
    >>> ctx.start()
    >>> # ... do work ...
    >>> ctx.stop()
    >>> ctx.duration_s  # doctest: +SKIP
    0.001
    """

    name: str
    start_time: float | None = None
    end_time: float | None = None

    def start(self) -> None:
        """Record start time."""
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        """Record end time."""
        self.end_time = time.perf_counter()

    @property
    def duration_s(self) -> float:
        """Return duration in seconds.

        Returns
        -------
        float
            Duration in seconds, or 0.0 if timing is incomplete.
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time


@contextmanager
def timed(name: str) -> Generator[TimingContext]:
    """Context manager for timing an operation.

    Parameters
    ----------
    name
        Name of the timed operation.

    Yields
    ------
    TimingContext
        Timing context with duration available after exit.

    Examples
    --------
    >>> with timed("my_operation") as t:
    ...     pass  # do work
    >>> t.duration_s >= 0
    True
    """
    ctx = TimingContext(name)
    ctx.start()
    try:
        yield ctx
    finally:
        ctx.stop()


@dataclass
class ExecutionTracker:
    """Track execution of multiple steps.

    Attributes
    ----------
    results
        List of step results in execution order.

    Examples
    --------
    >>> tracker = ExecutionTracker()
    >>> tracker.record(StepResult.success("step1", duration_s=1.0))
    >>> tracker.record(StepResult.success("step2", duration_s=0.5))
    >>> tracker.total_duration_s
    1.5
    """

    results: list[StepResult] = field(default_factory=list)

    def record(self, result: StepResult) -> None:
        """Record a step result.

        Parameters
        ----------
        result
            Step result to record.
        """
        self.results.append(result)

    @property
    def completed_steps(self) -> list[str]:
        """Return names of completed (success/skipped) steps.

        Returns
        -------
        list[str]
            Names of completed steps.
        """
        return [r.step_name for r in self.results if r.ok]

    @property
    def failed_steps(self) -> list[str]:
        """Return names of failed steps.

        Returns
        -------
        list[str]
            Names of failed steps.
        """
        return [r.step_name for r in self.results if not r.ok]

    @property
    def total_duration_s(self) -> float:
        """Return total duration of all steps.

        Returns
        -------
        float
            Total duration in seconds.
        """
        return sum(r.duration_s for r in self.results)

    @property
    def all_passed(self) -> bool:
        """Return True if all steps passed.

        Returns
        -------
        bool
            True if all steps succeeded or were skipped.
        """
        return all(r.ok for r in self.results)


__all__ = [
    "ExecutionTracker",
    "StepResult",
    "StepStatus",
    "TimingContext",
    "timed",
]
