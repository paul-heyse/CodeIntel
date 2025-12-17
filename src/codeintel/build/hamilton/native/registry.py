"""Convenience API for native Hamilton target discovery.

This module intentionally stays thin:

- Native module discovery is implemented in :mod:`codeintel.build.hamilton.native.discovery`.
- Target metadata is compiled by :func:`codeintel.build.target_catalog.load_target_specs`.

The functions here exist as stable import points for callers that need to:

- Load native modules for driver composition.
- Determine whether a target is part of the Hamilton-native catalog.
"""

from __future__ import annotations

from functools import lru_cache

from codeintel.build.hamilton.native.discovery import load_native_modules
from codeintel.build.target_catalog import load_target_specs


@lru_cache(maxsize=1)
def native_target_names() -> frozenset[str]:
    """Return the set of target names in the canonical Hamilton-native catalog.

    Returns
    -------
    frozenset[str]
        Set of target names registered in the native target catalog.
    """
    return frozenset(target.name for target in load_target_specs())


def is_native_target(target_name: str) -> bool:
    """Check if a target exists in the canonical Hamilton-native catalog.

    Parameters
    ----------
    target_name
        Build target name to check.

    Returns
    -------
    bool
        True if the target is registered as part of the native catalog.
    """
    return target_name in native_target_names()


__all__ = [
    "is_native_target",
    "load_native_modules",
    "native_target_names",
]
