"""Event protocol definitions.

This module defines the core protocols for event handling.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class EventProtocol(Protocol):
    """Protocol for events.

    Attributes
    ----------
    EVENT_TYPE
        Unique event type identifier.
    """

    EVENT_TYPE: ClassVar[str]


@runtime_checkable
class EventHandlerProtocol(Protocol):
    """Protocol for event handlers.

    Examples
    --------
    >>> class MyHandler:
    ...     def handle(self, event: Event) -> None:
    ...         print(f"Received: {event}")
    """

    def handle(self, event: EventProtocol) -> None:
        """Handle an event.

        Parameters
        ----------
        event
            Event to handle.
        """
        ...


__all__ = [
    "EventHandlerProtocol",
    "EventProtocol",
]
