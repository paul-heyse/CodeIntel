"""Module discovery for Hamilton runtime composition."""

from __future__ import annotations

from types import ModuleType

from codeintel.build.hamilton.native.discovery import load_native_modules, native_module_paths


def resolve_module_paths(*, include_planning: bool = True) -> tuple[str, ...]:
    """Return module import paths for runtime composition.

    Returns
    -------
    tuple[str, ...]
        Module import paths for runtime composition.
    """
    paths = native_module_paths()
    if include_planning:
        return paths
    return tuple(path for path in paths if ".planning." not in path)


def resolve_modules(*, include_planning: bool = True) -> tuple[ModuleType, ...]:
    """Return imported modules for runtime composition.

    Returns
    -------
    tuple[ModuleType, ...]
        Imported runtime modules.
    """
    modules = load_native_modules()
    if include_planning:
        return modules
    return tuple(module for module in modules if ".planning." not in module.__name__)


__all__ = ["resolve_module_paths", "resolve_modules"]
