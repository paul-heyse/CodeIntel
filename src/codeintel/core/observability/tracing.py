"""Tracing utilities.

This module provides utilities for distributed tracing.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)


@dataclass
class Span:
    """A trace span representing an operation.

    Attributes
    ----------
    name
        Operation name.
    trace_id
        Unique trace identifier.
    span_id
        Unique span identifier.
    parent_id
        Parent span identifier, if any.
    attributes
        Span attributes.
    status
        Span status (ok, error).
    status_message
        Optional status message.
    start_time
        When the span started.
    end_time
        When the span ended.
    """

    name: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_id: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    status: str = "ok"
    status_message: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def set_attribute(self, key: str, value: object) -> None:
        """Set a span attribute.

        Parameters
        ----------
        key
            Attribute key.
        value
            Attribute value.
        """
        self.attributes[key] = value

    def set_status(self, status: str, message: str | None = None) -> None:
        """Set the span status.

        Parameters
        ----------
        status
            Status code (ok, error).
        message
            Optional status message.
        """
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time()

    @property
    def duration_s(self) -> float | None:
        """Get span duration in seconds.

        Returns
        -------
        float | None
            Duration if ended, None otherwise.
        """
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_ended(self) -> bool:
        """Check if span has ended.

        Returns
        -------
        bool
            True if ended.
        """
        return self.end_time is not None

    def __enter__(self) -> Self:
        """Enter context manager.

        Returns
        -------
        Self
            The span.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, ending span."""
        if exc_val is not None:
            self.set_status("error", str(exc_val))
        self.end()


class InMemoryTracer:
    """In-memory tracer for testing and development.

    Stores spans in memory for later inspection.

    Examples
    --------
    >>> tracer = InMemoryTracer()
    >>> with tracer.start_span("process_request") as span:
    ...     span.set_attribute("user_id", "123")
    ...     process_request()
    >>> len(tracer.spans)
    1
    """

    def __init__(self) -> None:
        """Initialize the tracer."""
        self._spans: list[Span] = []
        self._current_span: Span | None = None

    def start_span(self, name: str, **attributes: object) -> Span:
        """Start a new trace span.

        Parameters
        ----------
        name
            Span name.
        **attributes
            Initial span attributes.

        Returns
        -------
        Span
            The started span.
        """
        parent_id = self._current_span.span_id if self._current_span else None
        trace_id = self._current_span.trace_id if self._current_span else str(uuid.uuid4())

        span = Span(
            name=name,
            trace_id=trace_id,
            parent_id=parent_id,
            attributes=dict(attributes),
        )

        self._spans.append(span)
        self._current_span = span

        return span

    @property
    def spans(self) -> list[Span]:
        """Get all recorded spans.

        Returns
        -------
        list[Span]
            List of spans.
        """
        return list(self._spans)

    @property
    def current_span(self) -> Span | None:
        """Get the current active span.

        Returns
        -------
        Span | None
            Current span or None.
        """
        return self._current_span

    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
        self._current_span = None


@contextmanager
def trace_operation(
    tracer: InMemoryTracer,
    name: str,
    **attributes: object,
) -> Iterator[Span]:
    """Context manager for tracing operations.

    Parameters
    ----------
    tracer
        Tracer instance.
    name
        Operation name.
    **attributes
        Span attributes.

    Yields
    ------
    Span
        The active span.

    Examples
    --------
    >>> with trace_operation(tracer, "process", user="admin") as span:
    ...     result = do_work()
    ...     span.set_attribute("result_size", len(result))
    """
    span = tracer.start_span(name, **attributes)
    try:
        yield span
    except Exception as e:
        span.set_status("error", str(e))
        raise
    finally:
        span.end()


__all__ = [
    "InMemoryTracer",
    "Span",
    "trace_operation",
]
