"""Native target registry for Hamilton Phase 3.

This module provides functions to query native Hamilton targets that have been
migrated from plugin wrappers to pure Hamilton pipelines.

All native targets are implemented in a small set of domain modules under
``codeintel.build.hamilton.native``. This module provides a convenience API
for loading those modules for driver composition.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.registry import ALL_TARGETS

if TYPE_CHECKING:
    from types import ModuleType


_NATIVE_MODULE_PATHS: tuple[str, ...] = (
    "codeintel.build.hamilton.native.ingestion",
    "codeintel.build.hamilton.native.graphs",
    "codeintel.build.hamilton.native.analytics",
    "codeintel.build.hamilton.native.export",
)


def native_target_names() -> frozenset[str]:
    """Return the set of target names that have native implementations.

    Returns
    -------
    frozenset[str]
        Set of target names registered as native.

    Examples
    --------
    >>> names = native_target_names()
    >>> "risk_factors" in names
    True
    """
    return frozenset(t.name for t in ALL_TARGETS)


@lru_cache(maxsize=1)
def load_native_modules() -> tuple[ModuleType, ...]:
    """Load all native target modules for driver composition.

    Returns
    -------
    tuple[ModuleType, ...]
        Tuple of imported native target modules.

    Examples
    --------
    >>> modules = load_native_modules()
    >>> len(modules) > 0
    True
    """
    return tuple(importlib.import_module(module_path) for module_path in _NATIVE_MODULE_PATHS)


def is_native_target(target_name: str) -> bool:
    """Check if a target has a native implementation.

    Parameters
    ----------
    target_name
        Build target name to check.

    Returns
    -------
    bool
        True if the target is registered as native.

    Examples
    --------
    >>> is_native_target("risk_factors")
    True
    >>> is_native_target("modules")
    True
    >>> is_native_target("goids")
    True
    """
    return target_name in native_target_names()


__all__ = [
    "is_native_target",
    "load_native_modules",
    "native_target_names",
]
