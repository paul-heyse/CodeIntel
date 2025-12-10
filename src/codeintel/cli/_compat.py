"""Compatibility utilities for CLI module deprecation.

This module provides helpers for emitting deprecation warnings when
legacy module paths are used.
"""

from __future__ import annotations

import warnings
from typing import Any


def emit_deprecation_warning(old_module: str, new_module: str) -> None:
    """Emit a deprecation warning for module relocation.

    Parameters
    ----------
    old_module
        The old module path being deprecated.
    new_module
        The new canonical module path to use.

    """
    warnings.warn(
        f"Importing from '{old_module}' is deprecated. "
        f"Use '{new_module}' instead. "
        "This compatibility shim will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3,
    )


def deprecated_reexport(
    old_module: str,
    new_module: str,
    names: list[str],
    module_globals: dict[str, Any],
) -> list[str]:
    """Set up deprecated re-exports from a new module location.

    This function imports all specified names from the new module and adds
    them to the old module's globals, emitting a deprecation warning.

    Parameters
    ----------
    old_module
        The old module path being deprecated.
    new_module
        The new canonical module path to import from.
    names
        List of names to re-export.
    module_globals
        The globals() dict of the old module.

    Returns
    -------
    list[str]
        The list of names for __all__.

    """
    import importlib

    emit_deprecation_warning(old_module, new_module)

    new_mod = importlib.import_module(new_module)
    for name in names:
        if hasattr(new_mod, name):
            module_globals[name] = getattr(new_mod, name)
        else:
            warnings.warn(
                f"Name '{name}' not found in '{new_module}'",
                ImportWarning,
                stacklevel=2,
            )

    return names
