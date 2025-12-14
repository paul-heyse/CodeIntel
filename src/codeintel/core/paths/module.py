"""Module path conversion utilities.

This module provides utilities for converting between file paths
and Python module names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def path_to_module(path: str | Path) -> str:
    """Convert a file path to a Python module name.

    Parameters
    ----------
    path
        File path (e.g., "src/package/module.py").

    Returns
    -------
    str
        Module name (e.g., "src.package.module").

    Examples
    --------
    >>> path_to_module("src/package/module.py")
    'src.package.module'
    >>> path_to_module("package/__init__.py")
    'package'
    """
    path_str = str(path).replace("\\", "/")

    if path_str.endswith(".py"):
        path_str = path_str[:-3]

    if path_str.endswith("/__init__"):
        path_str = path_str[:-9]

    return path_str.replace("/", ".")


def module_to_path(module: str, *, as_package: bool = False) -> str:
    """Convert a Python module name to a file path.

    Parameters
    ----------
    module
        Module name (e.g., "package.module").
    as_package
        If True, return path to __init__.py.

    Returns
    -------
    str
        File path.

    Examples
    --------
    >>> module_to_path("package.module")
    'package/module.py'
    >>> module_to_path("package", as_package=True)
    'package/__init__.py'
    """
    path = module.replace(".", "/")

    if as_package:
        return f"{path}/__init__.py"
    return f"{path}.py"


def is_package_path(path: str | Path) -> bool:
    """Check if a path represents a package.

    Parameters
    ----------
    path
        File path to check.

    Returns
    -------
    bool
        True if path is an __init__.py file.

    Examples
    --------
    >>> is_package_path("package/__init__.py")
    True
    >>> is_package_path("module.py")
    False
    """
    path_str = str(path).replace("\\", "/")
    return path_str.endswith("/__init__.py") or path_str == "__init__.py"


# Backward compatibility alias
relpath_to_module = path_to_module
"""Alias for path_to_module.

.. deprecated:: 1.0
    Use ``path_to_module`` instead.
"""


__all__ = [
    "is_package_path",
    "module_to_path",
    "path_to_module",
    "relpath_to_module",
]
