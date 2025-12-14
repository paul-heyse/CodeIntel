"""Event emitter implementation.

This module provides a simple event emitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


@dataclass
class Event:
    """A simple event.

    Attributes
    ----------
    event_type
        Type of the event.
    data
        Event data.
    """

    EVENT_TYPE: ClassVar[str] = "event"
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)


class EventEmitter:
    """Simple event emitter.

    Provides a pub/sub mechanism for event handling.

    Examples
    --------
    >>> emitter = EventEmitter()
    >>> def on_data(event: Event) -> None:
    ...     print(f"Received: {event.data}")
    >>> emitter.on("data", on_data)
    >>> emitter.emit("data", {"value": 42})
    Received: {'value': 42}
    """

    def __init__(self) -> None:
        """Initialize the emitter."""
        self._handlers: dict[str, list[Callable[[Event], None]]] = {}

    def on(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> Callable[[], None]:
        """Register an event handler.

        Parameters
        ----------
        event_type
            Type of event to handle.
        handler
            Handler function.

        Returns
        -------
        Callable[[], None]
            Function to unregister the handler.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)
        log.debug("Registered handler for %s", event_type)

        def unsubscribe() -> None:
            self.off(event_type, handler)

        return unsubscribe

    def off(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> bool:
        """Unregister an event handler.

        Parameters
        ----------
        event_type
            Type of event.
        handler
            Handler to remove.

        Returns
        -------
        bool
            True if handler was found and removed.
        """
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)
            log.debug("Unregistered handler for %s", event_type)
            return True
        return False

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> int:
        """Emit an event.

        Parameters
        ----------
        event_type
            Type of event.
        data
            Event data.

        Returns
        -------
        int
            Number of handlers invoked.
        """
        event = Event(event_type=event_type, data=data or {})
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                log.exception("Error in event handler for %s", event_type)

        return len(handlers)

    def clear(self, event_type: str | None = None) -> None:
        """Clear event handlers.

        Parameters
        ----------
        event_type
            Type to clear, or None for all.
        """
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)

    def has_handlers(self, event_type: str) -> bool:
        """Check if handlers exist for an event type.

        Parameters
        ----------
        event_type
            Type to check.

        Returns
        -------
        bool
            True if handlers exist.
        """
        return bool(self._handlers.get(event_type))


__all__ = [
    "Event",
    "EventEmitter",
]
