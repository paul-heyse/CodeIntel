"""Async progress streaming for CLI operations.

Provide utilities for streaming progress events during async
operation execution, with support for multiple output formats.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, TextIO

from codeintel.cli.async_types import ProgressEvent, ProgressState, StreamingResult
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable


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


class AsyncProgressRenderer:
    """Render progress events to output stream.

    Parameters
    ----------
    config
        Progress stream configuration.
    """

    #: Spinner frames for animation
    SPINNER_FRAMES: ClassVar[list[str]] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

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
        parts = []

        # Timestamp
        if self._config.show_timestamps:
            ts = event.timestamp.strftime("%H:%M:%S")
            parts.append(f"[{ts}]")

        # Spinner or progress bar
        if event.state == ProgressState.RUNNING:
            if event.progress is not None:
                # Progress bar
                percent = int(event.progress * 100)
                bar_width = 20
                filled = int(bar_width * event.progress)
                bar = "█" * filled + "░" * (bar_width - filled)
                parts.append(f"[{bar}] {percent:3d}%")
            elif self._config.show_spinner:
                # Spinner
                frame = self.SPINNER_FRAMES[self._spinner_idx]
                self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_FRAMES)
                parts.append(f"[{frame}]")

        # State indicator
        state_icons = {
            ProgressState.PENDING: "○",
            ProgressState.RUNNING: "●",
            ProgressState.PAUSED: "◐",
            ProgressState.COMPLETED: "✓",
            ProgressState.FAILED: "✗",
            ProgressState.CANCELLED: "⊘",
        }
        parts.append(state_icons.get(event.state, "?"))

        # Operation ID
        parts.append(event.operation_id)

        # Message
        if event.message:
            parts.append(f"- {event.message}")

        # Item counts
        if event.items_total is not None:
            items_done = event.items_completed or 0
            parts.append(f"({items_done}/{event.items_total})")

        line = " ".join(parts)

        # Clear previous line if terminal supports it
        if event.state == ProgressState.RUNNING and self._last_line_length > 0:
            self._config.output.write("\r" + " " * self._last_line_length + "\r")

        # Write new line
        if event.state in {ProgressState.COMPLETED, ProgressState.FAILED, ProgressState.CANCELLED}:
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
    renderer = AsyncProgressRenderer(config)
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

    # Initial progress
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
        # Process item (in executor for sync functions)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda idx=i: process_item(idx))
        results.append(result)

        # Emit progress on batch boundaries
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

    # Completion
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

    # Final result
    yield StreamingResult[list[T]](
        result=CliResult.ok(results),
    )


@dataclass
class ProgressTracker:
    """Track progress across multiple operations.

    Parameters
    ----------
    operation_id
        Root operation identifier.
    callback
        Callback for progress events.
    """

    operation_id: str
    callback: Callable[[ProgressEvent], None] | None = None
    _items_completed: int = field(default=0, init=False)
    _items_total: int = field(default=0, init=False)
    _started_at: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)

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
            operation_id=self.operation_id,
            state=ProgressState.RUNNING,
            progress=progress,
            message=message,
            items_completed=self._items_completed,
            items_total=self._items_total,
        )

        if self.callback is not None:
            self.callback(event)

        return event

    def complete(self, message: str = "Completed") -> ProgressEvent:
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
            operation_id=self.operation_id,
            state=ProgressState.COMPLETED,
            progress=1.0,
            message=message,
            items_completed=self._items_completed,
            items_total=self._items_total,
        )

        if self.callback is not None:
            self.callback(event)

        return event

    def fail(self, message: str = "Failed") -> ProgressEvent:
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
            operation_id=self.operation_id,
            state=ProgressState.FAILED,
            message=message,
            items_completed=self._items_completed,
            items_total=self._items_total,
        )

        if self.callback is not None:
            self.callback(event)

        return event


__all__ = [
    "AsyncProgressRenderer",
    "ProgressStreamConfig",
    "ProgressTracker",
    "progress_generator",
    "stream_progress",
]
