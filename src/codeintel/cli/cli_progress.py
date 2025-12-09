"""Progress reporting infrastructure for CLI operations.

Provide progress bars and status indicators for long-running operations
using rich.progress as the rendering backend.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Self

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

LOG = logging.getLogger(__name__)


@dataclass
class ProgressConfig:
    """Configuration for progress reporting.

    Parameters
    ----------
    enabled
        Whether progress bars are enabled.
    verbose
        Whether to show verbose progress information.
    refresh_rate
        Refresh rate in Hz for progress updates.
    """

    enabled: bool = True
    verbose: bool = False
    refresh_rate: float = 10.0


@dataclass
class ProgressTracker:
    """Track progress of long-running operations.

    This class provides a high-level interface for tracking progress
    that can be used independently of the rendering backend.

    Parameters
    ----------
    config
        Progress configuration.
    """

    config: ProgressConfig = field(default_factory=ProgressConfig)
    _tasks: dict[str, TaskID] = field(default_factory=dict, init=False, repr=False)
    _progress: Progress | None = field(default=None, init=False, repr=False)

    def _get_progress(self) -> Progress:
        """Get or create the progress instance.

        Returns
        -------
        Progress
            Rich progress instance.
        """
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                refresh_per_second=self.config.refresh_rate,
                disable=not self.config.enabled,
            )
        return self._progress

    def add_task(
        self,
        description: str,
        total: float | None = None,
        *,
        task_id: str | None = None,
    ) -> str:
        """Add a task to track.

        Parameters
        ----------
        description
            Task description.
        total
            Total number of steps (None for indeterminate).
        task_id
            Optional task identifier (auto-generated if not provided).

        Returns
        -------
        str
            Task identifier.
        """
        progress = self._get_progress()
        rich_task_id = progress.add_task(description, total=total or 100)

        # Generate task_id if not provided
        if task_id is None:
            task_id = f"task_{len(self._tasks)}"

        self._tasks[task_id] = rich_task_id
        return task_id

    def update(
        self,
        task_id: str,
        *,
        advance: float | None = None,
        completed: float | None = None,
        description: str | None = None,
    ) -> None:
        """Update task progress.

        Parameters
        ----------
        task_id
            Task identifier.
        advance
            Amount to advance progress by.
        completed
            Absolute completion value to set.
        description
            New description (if changing).
        """
        if task_id not in self._tasks:
            LOG.warning("Unknown task_id: %s", task_id)
            return

        progress = self._get_progress()
        rich_task_id = self._tasks[task_id]

        # Call update with explicit parameters to satisfy type checker
        progress.update(
            rich_task_id,
            advance=advance,
            completed=completed,
            description=description,
        )

    def complete(self, task_id: str) -> None:
        """Mark a task as complete.

        Parameters
        ----------
        task_id
            Task identifier.
        """
        if task_id not in self._tasks:
            return

        progress = self._get_progress()
        task = progress.tasks[self._tasks[task_id]]
        self.update(task_id, completed=task.total or 100)

    @contextmanager
    def task(
        self,
        description: str,
        total: float | None = None,
    ) -> Iterator[str]:
        """Context manager for a progress task.

        Parameters
        ----------
        description
            Task description.
        total
            Total number of steps.

        Yields
        ------
        str
            Task identifier for updates.
        """
        task_id = self.add_task(description, total=total)
        try:
            yield task_id
        finally:
            self.complete(task_id)

    def start(self) -> None:
        """Start the progress display."""
        if self.config.enabled:
            self._get_progress().start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self._progress is not None:
            self._progress.stop()

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns
        -------
        Self
            The progress tracker instance.
        """
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager."""
        self.stop()


# Global progress tracker
_PROGRESS_TRACKER: ProgressTracker | None = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker.

    Returns
    -------
    ProgressTracker
        Global progress tracker instance.
    """
    global _PROGRESS_TRACKER  # noqa: PLW0603
    if _PROGRESS_TRACKER is None:
        _PROGRESS_TRACKER = ProgressTracker()
    return _PROGRESS_TRACKER


def configure_progress(
    *,
    enabled: bool = True,
    verbose: bool = False,
) -> None:
    """Configure progress reporting.

    Parameters
    ----------
    enabled
        Whether progress bars are enabled.
    verbose
        Whether to show verbose progress information.
    """
    global _PROGRESS_TRACKER  # noqa: PLW0603
    _PROGRESS_TRACKER = ProgressTracker(
        config=ProgressConfig(enabled=enabled, verbose=verbose),
    )


@contextmanager
def progress_context(
    description: str,
    total: float | None = None,
    *,
    enabled: bool = True,
) -> Iterator[str]:
    """Context manager for progress tracking.

    Parameters
    ----------
    description
        Task description.
    total
        Total number of steps.
    enabled
        Whether progress is enabled.

    Yields
    ------
    str
        Task identifier for updates.
    """
    if not enabled:
        yield ""
        return

    tracker = get_progress_tracker()
    with tracker, tracker.task(description, total=total) as task_id:
        yield task_id


def iter_with_progress[T](
    items: list[T],
    description: str,
    *,
    enabled: bool = True,
) -> Iterator[T]:
    """Iterate over items with progress tracking.

    Parameters
    ----------
    items
        Items to iterate over.
    description
        Task description.
    enabled
        Whether progress is enabled.

    Yields
    ------
    T
        Items from the input list.
    """
    if not enabled:
        yield from items
        return

    tracker = get_progress_tracker()
    with tracker:
        task_id = tracker.add_task(description, total=len(items))
        for item in items:
            yield item
            tracker.update(task_id, advance=1)


__all__ = [
    "ProgressConfig",
    "ProgressTracker",
    "configure_progress",
    "get_progress_tracker",
    "iter_with_progress",
    "progress_context",
]
