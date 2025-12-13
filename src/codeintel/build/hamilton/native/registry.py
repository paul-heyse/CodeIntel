"""Native target registry for Hamilton Phase 3.

This module provides functions to query native Hamilton targets that have been
migrated from plugin wrappers to pure Hamilton pipelines.

All native target information is now stored in the UnifiedRegistry. This
module provides a convenience API that delegates to the unified registry.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def native_target_names() -> frozenset[str]:
    """Return the set of target names that have native implementations.

    This function delegates to the UnifiedRegistry.

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
    from codeintel.build.unified_registry import get_unified_registry  # noqa: PLC0415

    return get_unified_registry().native_target_names()


def load_native_modules() -> tuple[ModuleType, ...]:
    """Load all native target modules for driver composition.

    This function loads modules from the UnifiedRegistry's native_module
    registrations.

    Returns
    -------
    tuple[ModuleType, ...]
        Tuple of imported native target modules.

    Raises
    ------
    ImportError
        If a registered module cannot be imported.

    Examples
    --------
    >>> modules = load_native_modules()
    >>> len(modules) > 0
    True
    """
    from codeintel.build.unified_registry import get_unified_registry  # noqa: PLC0415

    registry = get_unified_registry()
    modules: list[ModuleType] = []

    for reg in registry.get_all_registrations():
        if reg.native_module is not None:
            try:
                module = importlib.import_module(reg.native_module)
                modules.append(module)
            except ImportError as e:
                msg = f"Failed to import native target module '{reg.native_module}': {e}"
                raise ImportError(msg) from e

    return tuple(modules)


def is_native_target(target_name: str) -> bool:
    """Check if a target has a native implementation.

    This function delegates to the UnifiedRegistry.

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
    False
    """
    from codeintel.build.unified_registry import get_unified_registry  # noqa: PLC0415

    return get_unified_registry().is_native_target(target_name)


__all__ = [
    "is_native_target",
    "load_native_modules",
    "native_target_names",
]
