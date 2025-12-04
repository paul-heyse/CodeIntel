"""Base functional plugin for wrapping callables as plugins.

This module provides `BaseFunctionalPlugin`, a generic base class for
creating plugins from functions. Both analytics and graph subsystems
use type aliases derived from this base.

Example
-------
>>> # Analytics uses:
>>> FunctionalPlugin = BaseFunctionalPlugin[PluginExecutionContext, PluginMetadata]
>>>
>>> # Graphs uses:
>>> FunctionalGraphPlugin = BaseFunctionalPlugin[GraphPluginExecutionContext, GraphPluginMetadata]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from codeintel.core.plugins.protocol import PluginMetadata, ValidationResult
from codeintel.core.plugins.result import PluginResult


@dataclass
class BaseFunctionalPlugin[TCtx, TMeta: PluginMetadata]:
    """Base class for function-wrapped plugins.

    Wraps a callable function as a plugin, providing metadata and validation.
    This class is generic over the context type and metadata type to support
    both analytics and graph plugins.

    Type Parameters
    ---------------
    TCtx
        The execution context type (e.g., PluginExecutionContext).
    TMeta
        The metadata type, must be PluginMetadata or a subclass.

    Attributes
    ----------
    _metadata
        Plugin metadata describing the plugin.
    _execute_fn
        The wrapped function that performs the plugin's work.
    _validate_fn
        Optional custom validation function.

    Example
    -------
    >>> def my_compute(ctx: PluginExecutionContext) -> PluginResult:
    ...     return PluginResult.ok()
    >>> plugin = BaseFunctionalPlugin(
    ...     _metadata=metadata,
    ...     _execute_fn=my_compute,
    ... )
    """

    _metadata: TMeta
    _execute_fn: Callable[[TCtx], PluginResult]
    _validate_fn: Callable[[TCtx], ValidationResult] | None = None

    @property
    def metadata(self) -> TMeta:
        """Return plugin metadata.

        Returns
        -------
        TMeta
            Metadata describing the plugin.
        """
        return self._metadata

    def execute(self, ctx: TCtx) -> PluginResult:
        """Execute the wrapped function.

        Parameters
        ----------
        ctx
            Plugin execution context.

        Returns
        -------
        PluginResult
            Result produced by the underlying callable.
        """
        return self._execute_fn(ctx)

    def validate_inputs(self, ctx: TCtx) -> ValidationResult:
        """Validate inputs using the custom validator or default.

        Parameters
        ----------
        ctx
            Plugin execution context.

        Returns
        -------
        ValidationResult
            Validation result from the custom validator or a default success.
        """
        if self._validate_fn is not None:
            return self._validate_fn(ctx)
        return ValidationResult.success()


__all__ = ["BaseFunctionalPlugin"]
