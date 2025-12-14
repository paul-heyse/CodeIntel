"""Hook registry for extensibility.

This module provides a registry for hook points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


@dataclass
class HookRegistry:
    """Registry for extension hooks.

    Allows registering callbacks at named hook points.

    Examples
    --------
    >>> hooks = HookRegistry()
    >>> hooks.register("pre_process", lambda ctx: print("Processing..."))
    >>> hooks.invoke("pre_process", {"data": "value"})
    Processing...
    """

    _hooks: dict[str, list[Callable[[dict[str, Any]], None]]] = field(default_factory=dict)

    def register(
        self,
        hook_name: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register a callback for a hook.

        Parameters
        ----------
        hook_name
            Name of the hook.
        callback
            Callback function.
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        self._hooks[hook_name].append(callback)
        log.debug("Registered hook: %s", hook_name)

    def invoke(
        self,
        hook_name: str,
        context: dict[str, Any] | None = None,
    ) -> int:
        """Invoke all callbacks for a hook.

        Parameters
        ----------
        hook_name
            Name of the hook.
        context
            Context data for callbacks.

        Returns
        -------
        int
            Number of callbacks invoked.
        """
        callbacks = self._hooks.get(hook_name, [])
        ctx = context or {}

        for callback in callbacks:
            try:
                callback(ctx)
            except Exception:
                log.exception("Error in hook callback: %s", hook_name)

        return len(callbacks)

    def clear(self, hook_name: str | None = None) -> None:
        """Clear hooks.

        Parameters
        ----------
        hook_name
            Hook to clear, or None for all.
        """
        if hook_name is None:
            self._hooks.clear()
        else:
            self._hooks.pop(hook_name, None)

    def has_hooks(self, hook_name: str) -> bool:
        """Check if hooks are registered.

        Parameters
        ----------
        hook_name
            Hook to check.

        Returns
        -------
        bool
            True if hooks exist.
        """
        return bool(self._hooks.get(hook_name))


__all__ = [
    "HookRegistry",
]
