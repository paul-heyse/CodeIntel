"""Incremental execution traits for plugins supporting partial updates.

This module provides protocols for plugins that can determine if they
need to run based on input changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IncrementalPlugin(Protocol):
    """Trait for plugins that support incremental execution.

    Plugins implementing this trait can determine if they need to
    run based on input changes and can produce partial results.

    This trait uses `object` for context type to allow domain-specific
    context types (PluginExecutionContext, IngestExecutionContext, etc.)
    to be used in implementations.

    Example
    -------
    >>> class MyIncrementalPlugin(BasePlugin):
    ...     def compute_input_hash(self, ctx: PluginExecutionContext) -> str:
    ...         return hashlib.md5(ctx.repo.encode()).hexdigest()
    ...
    ...     def is_unchanged(self, ctx: PluginExecutionContext, prior_hash: str | None) -> bool:
    ...         return prior_hash == self.compute_input_hash(ctx)
    """

    def compute_input_hash(self, ctx: object) -> str:
        """Compute a hash of the plugin's inputs.

        Parameters
        ----------
        ctx
            Execution context (domain-specific type).

        Returns
        -------
        str
            Hash of inputs for change detection.
        """
        ...

    def is_unchanged(self, ctx: object, prior_hash: str | None) -> bool:
        """Check if inputs have changed since last run.

        Parameters
        ----------
        ctx
            Execution context (domain-specific type).
        prior_hash
            Hash from prior execution.

        Returns
        -------
        bool
            True if inputs are unchanged.
        """
        ...


def is_incremental(plugin: object) -> bool:
    """Check if a plugin implements IncrementalPlugin.

    Parameters
    ----------
    plugin
        Plugin to check.

    Returns
    -------
    bool
        True if plugin supports incremental execution.
    """
    return isinstance(plugin, IncrementalPlugin)


__all__ = [
    "IncrementalPlugin",
    "is_incremental",
]
