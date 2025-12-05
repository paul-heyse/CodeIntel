"""Progress reporting traits for long-running plugins.

This module provides protocols and mixins for plugins that report
execution progress.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressReportingPlugin(Protocol):
    """Trait for plugins that report execution progress.

    Plugins implementing this trait can provide progress updates
    during long-running operations, enabling progress bars and
    status displays.

    Example
    -------
    >>> class LongRunningPlugin(BasePlugin, ProgressReportingPlugin):
    ...     def set_progress_callback(
    ...         self,
    ...         callback: Callable[[float, str], None],
    ...     ) -> None:
    ...         self._callback = callback
    """

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None],
    ) -> None:
        """Set a callback for progress reporting.

        Parameters
        ----------
        callback
            Callback receiving progress (0-1) and status message.
        """
        ...


class ProgressReportingMixin:
    """Mixin providing progress reporting to plugins.

    Use this mixin to implement ProgressReportingPlugin with
    built-in progress callback management.

    Example
    -------
    >>> class MyPlugin(BasePlugin, ProgressReportingMixin):
    ...     def compute(self, ctx):
    ...         for i, item in enumerate(items):
    ...             self.report_progress(i / len(items), f"Processing {item}")
    ...             process(item)
    """

    _progress_callback: Callable[[float, str], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[float, str], None],
    ) -> None:
        """Set the progress reporting callback.

        Parameters
        ----------
        callback
            Function receiving progress (0-1) and status message.
        """
        self._progress_callback = callback

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report execution progress.

        Parameters
        ----------
        progress
            Progress value between 0.0 and 1.0.
        message
            Optional status message describing current operation.
        """
        if self._progress_callback is not None:
            self._progress_callback(progress, message)


def is_progress_reporting(plugin: object) -> bool:
    """Check if a plugin implements ProgressReportingPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin reports progress.
    """
    return isinstance(plugin, ProgressReportingPlugin)


__all__ = [
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "is_progress_reporting",
]
