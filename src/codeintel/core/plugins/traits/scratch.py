"""Scratch store access traits for plugin data sharing.

This module provides protocols and mixins for plugins that share data
via the scratch store mechanism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codeintel.core.plugins.execution.context import PluginScratch


class ScratchContext(Protocol):
    """Protocol for contexts that provide scratch store access.

    This protocol enables type-safe access to the scratch store
    without depending on a specific context implementation.
    """

    @property
    def scratch(self) -> PluginScratch:
        """Return the scratch store.

        Returns
        -------
        PluginScratch
            Scratch store for inter-plugin communication.
        """
        ...


class WithDependencyData:
    """Mixin for plugins that consume data from dependent plugins.

    Enable type-safe access to data populated by upstream plugins
    via the scratch store. This mixin provides a consistent interface
    across all plugin domains (analytics, graphs, ingestion).

    Example
    -------
    >>> class ConsumerPlugin(BasePlugin, WithDependencyData):
    ...     def compute(self, ctx):
    ...         # Get data from upstream plugin
    ...         metrics = self.get_dependency_data(ctx, "function_metrics")
    ...         if metrics is None:
    ...             return PluginResult.fail("Missing function metrics")
    ...         # Use metrics...
    """

    @staticmethod
    def get_dependency_data[T](
        ctx: ScratchContext,
        key: str,
        default: T | None = None,
    ) -> T | None:
        """Retrieve data populated by a dependent plugin.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        key
            Key used by the upstream plugin.
        default
            Default value if not found.

        Returns
        -------
        T | None
            Data from upstream plugin or default.
        """
        return ctx.scratch.consume(key, default)

    @staticmethod
    def set_dependency_data(
        ctx: ScratchContext,
        key: str,
        value: object,
    ) -> None:
        """Store data for downstream plugins.

        Parameters
        ----------
        ctx
            Execution context with scratch store.
        key
            Key for downstream access.
        value
            Data to store.
        """
        ctx.scratch.declare(key, value)


__all__ = [
    "ScratchContext",
    "WithDependencyData",
]
