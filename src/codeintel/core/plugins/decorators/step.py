"""Plugin decorator factory for creating domain-specific plugin decorators.

This module provides factory functions for creating plugins from decorated
functions. The shared logic is used by both analytics and graph plugin
decorators.

Example
-------
>>> # In analytics/core/registry.py
>>> plugin_instance = make_plugin_instance(
...     fn=my_function,
...     options=options,
...     plugin_factory=lambda meta, fn: FunctionalPlugin(_metadata=meta, _execute_fn=fn),
...     to_metadata=lambda opts, f: opts.to_metadata(f),
...     register_fn=registry.register,
... )
"""

from __future__ import annotations

from collections.abc import Callable

from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.core.plugins.types.result import PluginResult


def make_plugin_instance[TCtx, TMeta: PluginMetadata, TPlugin, TOptions](
    fn: Callable[[TCtx], PluginResult],
    options: TOptions,
    plugin_factory: Callable[[TMeta, Callable[[TCtx], PluginResult]], TPlugin],
    to_metadata: Callable[[TOptions, Callable[[TCtx], PluginResult]], TMeta],
    register_fn: Callable[[TPlugin], None] | None = None,
) -> TPlugin:
    """Create a plugin instance from a function and options.

    This is the core factory logic shared by all plugin decorators.
    It converts options to metadata, creates the plugin instance,
    and optionally registers it.

    Parameters
    ----------
    fn
        The function to wrap as a plugin.
    options
        Plugin options containing metadata fields.
    plugin_factory
        Factory function that creates the plugin instance from metadata and fn.
    to_metadata
        Function that converts options to metadata.
    register_fn
        Optional function to register the plugin after creation.

    Returns
    -------
    TPlugin
        The created plugin instance.

    Example
    -------
    >>> plugin = make_plugin_instance(
    ...     fn=compute_metrics,
    ...     options=PluginMetaOptions(name="metrics"),
    ...     plugin_factory=lambda m, f: FunctionalPlugin(_metadata=m, _execute_fn=f),
    ...     to_metadata=lambda o, f: o.to_metadata(f),
    ...     register_fn=registry.register,
    ... )
    """
    metadata = to_metadata(options, fn)
    plugin_instance = plugin_factory(metadata, fn)

    if register_fn is not None:
        register_fn(plugin_instance)

    return plugin_instance


__all__ = ["make_plugin_instance"]
