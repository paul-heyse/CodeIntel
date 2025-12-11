"""Unified progress tracking for CLI operations.

Provide progress tracking infrastructure that works for both
sync (callbacks) and async (generators) execution modes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, Self, TextIO

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from codeintel.cli.core import CliResult
from codeintel.cli.execution.types import (
    ProgressConfig,
    ProgressEvent,
    ProgressState,
    StreamingResult,
)
from codeintel.core.singleton import SingletonHolder

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterator

    from rich.progress import (
        TaskID,
    )


LOG = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    """Progress tracker supporting sync callbacks and async generators.

    Use for both sync (via callbacks) and async (via stream) modes.

    Parameters
    ----------
    config
        Progress configuration.
    """

    config: ProgressConfig = field(default_factory=ProgressConfig)
    _tasks: dict[str, TaskID] = field(default_factory=dict, init=False, repr=False)
    _progress: Progress | None = field(default=None, init=False, repr=False)
    _callbacks: list[Callable[[ProgressEvent], None]] = field(
        default_factory=list, init=False, repr=False
    )
    _queue: asyncio.Queue[ProgressEvent] | None = field(default=None, init=False, repr=False)
    _operation_id: str = field(default="", init=False, repr=False)
    _items_completed: int = field(default=0, init=False)
    _items_total: int = field(default=0, init=False)
    _started_at: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)

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
                refresh_per_second=1.0 / self.config.update_interval,
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

        progress.update(
            rich_task_id,
            advance=advance,
            completed=completed,
            description=description,
        )

        # Also emit event if queue is available
        if self._queue is not None:
            total_val = progress.tasks[rich_task_id].total or 100
            current_completed = progress.tasks[rich_task_id].completed
            event = ProgressEvent(
                operation_id=self._operation_id or task_id,
                state=ProgressState.RUNNING,
                progress=current_completed / total_val if total_val > 0 else None,
                message=description or "",
                items_completed=int(current_completed),
                items_total=int(total_val),
            )
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)

        # Call callbacks
        for callback in self._callbacks:
            total_val = progress.tasks[rich_task_id].total or 100
            current_completed = progress.tasks[rich_task_id].completed
            event = ProgressEvent(
                operation_id=self._operation_id or task_id,
                state=ProgressState.RUNNING,
                progress=current_completed / total_val if total_val > 0 else None,
                message=description or "",
                items_completed=int(current_completed),
                items_total=int(total_val),
            )
            callback(event)

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

    def add_callback(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Add sync callback for progress updates.

        Parameters
        ----------
        callback
            Callback function.
        """
        self._callbacks.append(callback)

    def set_total(self, total: int) -> None:
        """Set total items count.

        Parameters
        ----------
        total
            Total number of items.
        """
        self._items_total = total

    def increment(self, message: str = "") -> ProgressEvent:
        """Increment progress and emit event.

        Parameters
        ----------
        message
            Status message.

        Returns
        -------
        ProgressEvent
            The emitted event.
        """
        self._items_completed += 1
        progress = self._items_completed / self._items_total if self._items_total > 0 else None

        event = ProgressEvent(
            operation_id=self._operation_id,
            state=ProgressState.RUNNING,
            progress=progress,
            message=message,
            items_completed=self._items_completed,
            items_total=self._items_total,
        )

        for callback in self._callbacks:
            callback(event)

        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)

        return event

    def mark_complete(self, message: str = "Completed") -> ProgressEvent:
        """Mark operation as complete.

        Parameters
        ----------
        message
            Completion message.

        Returns
        -------
        ProgressEvent
            Completion event.
        """
        event = ProgressEvent(
            operation_id=self._operation_id,
            state=ProgressState.COMPLETED,
            progress=1.0,
            message=message,
            items_completed=self._items_completed,
            items_total=self._items_total,
        )

        for callback in self._callbacks:
            callback(event)

        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)

        return event

    def mark_failed(self, message: str = "Failed") -> ProgressEvent:
        """Mark operation as failed.

        Parameters
        ----------
        message
            Failure message.

        Returns
        -------
        ProgressEvent
            Failure event.
        """
        event = ProgressEvent(
            operation_id=self._operation_id,
            state=ProgressState.FAILED,
            message=message,
            items_completed=self._items_completed,
            items_total=self._items_total,
        )

        for callback in self._callbacks:
            callback(event)

        if self._queue is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)

        return event

    async def stream(self) -> AsyncGenerator[ProgressEvent]:
        """Stream progress events (async).

        Yields
        ------
        ProgressEvent
            Progress events as they occur.
        """
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=100)

        while True:
            event = await self._queue.get()
            yield event
            if event.state in {
                ProgressState.COMPLETED,
                ProgressState.FAILED,
                ProgressState.CANCELLED,
            }:
                break


@dataclass
class ProgressStreamConfig:
    """Configuration for progress streaming.

    Parameters
    ----------
    output
        Output stream for progress events.
    format
        Output format ('text', 'json', 'jsonl').
    show_timestamps
        Include timestamps in output.
    show_spinner
        Show spinner animation for indeterminate progress.
    quiet
        Suppress progress output (only show result).
    """

    output: TextIO = field(default_factory=lambda: sys.stderr)
    format: str = "text"
    show_timestamps: bool = True
    show_spinner: bool = True
    quiet: bool = False


class ProgressRenderer:
    """Render progress events to console.

    Work for both sync and async execution modes.

    Parameters
    ----------
    config
        Progress stream configuration.
    """

    SPINNER_FRAMES: ClassVar[list[str]] = [
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    ]

    def __init__(self, config: ProgressStreamConfig | None = None) -> None:
        """Initialize progress renderer."""
        self._config = config or ProgressStreamConfig()
        self._spinner_idx = 0
        self._last_line_length = 0

    def render(self, event: ProgressEvent) -> None:
        """Render a progress event.

        Parameters
        ----------
        event
            Progress event to render.
        """
        if self._config.quiet:
            return

        if self._config.format == "json":
            self._render_json(event)
        elif self._config.format == "jsonl":
            self._render_jsonl(event)
        else:
            self._render_text(event)

    def _render_text(self, event: ProgressEvent) -> None:
        """Render event as text.

        Parameters
        ----------
        event
            Progress event.
        """
        parts: list[str] = []

        if self._config.show_timestamps:
            timestamp = event.timestamp.strftime("%H:%M:%S")
            parts.append(f"[{timestamp}]")

        if event.state == ProgressState.RUNNING:
            if event.progress is not None:
                percent = int(event.progress * 100)
                bar_width = 20
                filled = int(bar_width * event.progress)
                bar = "█" * filled + "░" * (bar_width - filled)
                parts.append(f"[{bar}] {percent:3d}%")
            elif self._config.show_spinner:
                frame = self.SPINNER_FRAMES[self._spinner_idx]
                self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_FRAMES)
                parts.append(f"[{frame}]")

        state_icons = {
            ProgressState.PENDING: "○",
            ProgressState.RUNNING: "●",
            ProgressState.PAUSED: "◐",
            ProgressState.COMPLETED: "✓",
            ProgressState.FAILED: "✗",
            ProgressState.CANCELLED: "⊘",
        }
        parts.append(state_icons.get(event.state, "?"))
        parts.append(event.operation_id)

        if event.message:
            parts.append(f"- {event.message}")

        if event.items_total is not None:
            items_done = event.items_completed or 0
            parts.append(f"({items_done}/{event.items_total})")

        line = " ".join(parts)

        if event.state == ProgressState.RUNNING and self._last_line_length > 0:
            self._config.output.write("\r" + " " * self._last_line_length + "\r")

        if event.state in {
            ProgressState.COMPLETED,
            ProgressState.FAILED,
            ProgressState.CANCELLED,
        }:
            self._config.output.write(line + "\n")
            self._last_line_length = 0
        else:
            self._config.output.write(line)
            self._last_line_length = len(line)

        self._config.output.flush()

    def _render_json(self, event: ProgressEvent) -> None:
        """Render event as pretty JSON.

        Parameters
        ----------
        event
            Progress event.
        """
        json_str = json.dumps(event.to_dict(), indent=2)
        self._config.output.write(json_str + "\n")
        self._config.output.flush()

    def _render_jsonl(self, event: ProgressEvent) -> None:
        """Render event as JSON Lines.

        Parameters
        ----------
        event
            Progress event.
        """
        json_str = json.dumps(event.to_dict())
        self._config.output.write(json_str + "\n")
        self._config.output.flush()

    def clear(self) -> None:
        """Clear any in-progress line."""
        if self._last_line_length > 0:
            self._config.output.write("\r" + " " * self._last_line_length + "\r")
            self._config.output.flush()
            self._last_line_length = 0


async def stream_progress[T](
    source: AsyncGenerator[StreamingResult[T]],
    *,
    config: ProgressStreamConfig | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
) -> T | None:
    """Stream progress events from an async generator.

    Render progress events and return the final result.

    Parameters
    ----------
    source
        Async generator yielding streaming results.
    config
        Progress stream configuration.
    on_progress
        Optional callback for each progress event.

    Returns
    -------
    T | None
        Final result data, or None if operation failed.

    Raises
    ------
    asyncio.CancelledError
        If the stream is cancelled.
    """
    renderer = ProgressRenderer(config)
    final_result: T | None = None

    try:
        async for item in source:
            if item.is_progress and item.progress is not None:
                renderer.render(item.progress)
                if on_progress is not None:
                    on_progress(item.progress)

            elif item.is_result and item.result is not None:
                renderer.clear()
                if item.result.success:
                    final_result = item.result.data

    except asyncio.CancelledError:
        renderer.clear()
        cancelled_event = ProgressEvent(
            operation_id="stream",
            state=ProgressState.CANCELLED,
            message="Stream cancelled",
        )
        renderer.render(cancelled_event)
        raise

    return final_result


async def progress_generator[T](
    operation_id: str,
    total_items: int,
    process_item: Callable[[int], T],
    *,
    batch_size: int = 1,
) -> AsyncGenerator[StreamingResult[list[T]]]:
    """Create a progress-reporting async generator.

    Parameters
    ----------
    operation_id
        Operation identifier.
    total_items
        Total number of items to process.
    process_item
        Function to process each item.
    batch_size
        Number of items per progress update.

    Yields
    ------
    StreamingResult[list[T]]
        Progress events and final result.
    """
    results: list[T] = []

    yield StreamingResult[list[T]](
        progress=ProgressEvent(
            operation_id=operation_id,
            state=ProgressState.RUNNING,
            progress=0.0,
            message=f"Processing 0/{total_items} items",
            items_completed=0,
            items_total=total_items,
        ),
    )

    for items_completed, i in enumerate(range(total_items), 1):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda idx=i: process_item(idx))
        results.append(result)

        if items_completed % batch_size == 0 or items_completed == total_items:
            progress = items_completed / total_items
            yield StreamingResult[list[T]](
                progress=ProgressEvent(
                    operation_id=operation_id,
                    state=ProgressState.RUNNING,
                    progress=progress,
                    message=f"Processing {items_completed}/{total_items} items",
                    items_completed=items_completed,
                    items_total=total_items,
                ),
            )

    yield StreamingResult[list[T]](
        progress=ProgressEvent(
            operation_id=operation_id,
            state=ProgressState.COMPLETED,
            progress=1.0,
            message=f"Completed {total_items} items",
            items_completed=total_items,
            items_total=total_items,
        ),
    )

    yield StreamingResult[list[T]](
        result=CliResult.ok(results),
    )


class ProgressTrackerHolder(SingletonHolder[ProgressTracker]):
    """Singleton holder for the progress tracker."""


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker.

    Returns
    -------
    ProgressTracker
        Global progress tracker instance.
    """
    return ProgressTrackerHolder.get(ProgressTracker)


def configure_progress(
    *,
    enabled: bool = True,
    show_spinner: bool = True,
    update_interval: float = 0.1,
) -> None:
    """Configure progress reporting.

    Parameters
    ----------
    enabled
        Whether progress is enabled.
    show_spinner
        Whether to show spinner.
    update_interval
        Minimum update interval in seconds.
    """
    ProgressTrackerHolder.reset()
    ProgressTrackerHolder.get(
        lambda: ProgressTracker(
            config=ProgressConfig(
                enabled=enabled,
                show_spinner=show_spinner,
                update_interval=update_interval,
            ),
        )
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
    "ProgressRenderer",
    "ProgressStreamConfig",
    "ProgressTracker",
    "configure_progress",
    "get_progress_tracker",
    "iter_with_progress",
    "progress_context",
    "progress_generator",
    "stream_progress",
]
