"""Isolation traits for plugins requiring process/thread separation.

This module provides the IsolatedPlugin protocol for plugins that
need to run in isolated processes or threads.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class IsolatedPlugin(Protocol):
    """Trait for plugins requiring process or thread isolation.

    Plugins implementing this trait will be executed in a separate
    process or thread to prevent interference with other plugins.

    This is useful for plugins that:
    - Use libraries with global state
    - Need memory isolation
    - Risk crashing the process

    The "none" option is available for plugins that declare isolation
    capability but don't require it in certain configurations.

    Example
    -------
    >>> class UnsafePlugin(BasePlugin, IsolatedPlugin):
    ...     @property
    ...     def isolation_kind(self) -> Literal["process", "thread", "none"]:
    ...         return "process"  # Run in separate process
    """

    @property
    def isolation_kind(self) -> Literal["process", "thread", "none"]:
        """Return the isolation type required.

        Returns
        -------
        Literal["process", "thread", "none"]
            Type of isolation needed. "none" means no isolation required.
        """
        ...


def is_isolated(plugin: object) -> bool:
    """Check if a plugin implements IsolatedPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin requires isolation.
    """
    return isinstance(plugin, IsolatedPlugin)


__all__ = [
    "IsolatedPlugin",
    "is_isolated",
]
