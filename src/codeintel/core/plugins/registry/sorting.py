"""Shared plugin sorting and dependency resolution utilities.

This module provides domain-agnostic utilities for topological sorting
and dependency resolution that can be used by all plugin registries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

log = logging.getLogger(__name__)


# =============================================================================
# Protocol for plugins with capability metadata
# =============================================================================


class CapabilityProvider(Protocol):
    """Protocol for objects that provide capabilities.

    Used by the provider index builder to extract capability information
    from plugins without requiring a specific metadata structure.
    """

    @property
    def provides(self) -> tuple[str, ...]:
        """Return capabilities this plugin provides.

        Returns
        -------
        tuple[str, ...]
            Capability names provided by this plugin.
        """
        ...


# =============================================================================
# Sorting Utilities
# =============================================================================


def topological_sort[T](
    selected: dict[str, T],
    dependencies: dict[str, set[str]],
) -> list[T]:
    """Perform topological sort with cycle detection.

    Use iterative DFS with explicit stack to detect cycles and order plugins.
    This is a domain-agnostic utility that works with any type of plugin.

    Parameters
    ----------
    selected
        Selected plugins keyed by name.
    dependencies
        Dependency graph mapping plugin name to set of dependency names.

    Returns
    -------
    list[T]
        Plugins ordered based on dependencies (dependencies first).

    Raises
    ------
    ValueError
        If a dependency cycle is detected.

    Examples
    --------
    >>> plugins = {"a": "plugin_a", "b": "plugin_b", "c": "plugin_c"}
    >>> deps = {"a": set(), "b": {"a"}, "c": {"b"}}
    >>> topological_sort(plugins, deps)
    ['plugin_a', 'plugin_b', 'plugin_c']
    """
    ordered: list[T] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    # Iterative DFS with explicit stack
    for start in selected:
        if start in permanent:
            continue

        # Stack entries: (name, deps_list, is_processing)
        stack: list[tuple[str, list[str], bool]] = [
            (start, list(dependencies.get(start, set())), False)
        ]

        while stack:
            name, deps, processing = stack.pop()

            if processing:
                # Finished processing all deps, mark permanent
                temporary.discard(name)
                permanent.add(name)
                ordered.append(selected[name])
                continue

            if name in permanent:
                continue

            if name in temporary:
                # Cycle detected
                message = f"Dependency cycle detected involving plugin: {name}"
                raise ValueError(message)

            temporary.add(name)
            # Push back with processing=True to finalize after deps
            stack.append((name, [], True))

            # Push dependencies to process (filter for unvisited)
            unvisited_deps = [
                (dep, list(dependencies.get(dep, set())), False)
                for dep in deps
                if dep not in permanent
            ]
            stack.extend(unvisited_deps)

    return ordered


def build_provider_index[T: CapabilityProvider](
    selected: Mapping[str, T],
) -> dict[str, set[str]]:
    """Build index of capability -> provider plugin names.

    Create a mapping from capability names to the set of plugin names
    that provide each capability. Uses the `CapabilityProvider` protocol
    to access the `provides` property on each plugin.

    Parameters
    ----------
    selected
        Selected plugins to index, keyed by name.

    Returns
    -------
    dict[str, set[str]]
        Mapping of capability name to provider plugin names.

    Examples
    --------
    >>> class Plugin:
    ...     def __init__(self, provides):
    ...         self._provides = provides
    ...
    ...     @property
    ...     def provides(self):
    ...         return self._provides
    >>> plugins = {"a": Plugin(("cap1",)), "b": Plugin(("cap1", "cap2"))}
    >>> index = build_provider_index(plugins)
    >>> sorted(index["cap1"])
    ['a', 'b']
    >>> list(index["cap2"])
    ['b']
    """
    index: dict[str, set[str]] = {}
    for name, plugin in selected.items():
        for cap_name in plugin.provides:
            index.setdefault(cap_name, set()).add(name)
    return index


def build_provider_index_from_metadata[T](
    selected: Mapping[str, T],
    get_provides: Callable[[T], tuple[str, ...]],
) -> dict[str, set[str]]:
    """Build index of capability -> provider plugin names using accessor.

    Create a mapping from capability names to the set of plugin names
    that provide each capability, using a custom accessor function.

    Parameters
    ----------
    selected
        Selected plugins to index, keyed by name.
    get_provides
        Function to extract provides from plugin (e.g., lambda p: p.metadata.provides).

    Returns
    -------
    dict[str, set[str]]
        Mapping of capability name to provider plugin names.

    Examples
    --------
    >>> class Plugin:
    ...     def __init__(self, provides):
    ...         self.meta = type("Meta", (), {"provides": provides})()
    >>> plugins = {"a": Plugin(("cap1",)), "b": Plugin(("cap1", "cap2"))}
    >>> index = build_provider_index_from_metadata(plugins, lambda p: p.meta.provides)
    >>> sorted(index["cap1"])
    ['a', 'b']
    """
    index: dict[str, set[str]] = {}
    for name, plugin in selected.items():
        for cap_name in get_provides(plugin):
            index.setdefault(cap_name, set()).add(name)
    return index


__all__ = [
    "CapabilityProvider",
    "build_provider_index",
    "build_provider_index_from_metadata",
    "topological_sort",
]
