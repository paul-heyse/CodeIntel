"""Unified event infrastructure.

This module provides event-based communication patterns.
"""

from codeintel.core.events.emitter import (
    Event,
    EventEmitter,
)
from codeintel.core.events.protocol import (
    EventHandlerProtocol,
    EventProtocol,
)
from codeintel.core.events.registry import (
    HookRegistry,
)

__all__ = [
    "Event",
    "EventEmitter",
    "EventHandlerProtocol",
    "EventProtocol",
    "HookRegistry",
]
