"""Discovery utilities for native Hamilton target modules.

Native target implementations live under:

- ``codeintel.build.hamilton.native.ingestion``
- ``codeintel.build.hamilton.native.graphs``
- ``codeintel.build.hamilton.native.analytics``
- ``codeintel.build.hamilton.native.export``
- ``codeintel.build.hamilton.native.views``

This module provides deterministic module discovery without maintaining a
hand-edited list of module paths.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_NATIVE_PACKAGE_PREFIX = "codeintel.build.hamilton.native"

_NATIVE_DOMAINS: tuple[str, ...] = (
    "ingestion",
    "graphs",
    "analytics",
    "export",
    "planning",
    "views",
)


def _native_root_dir() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def native_module_paths() -> tuple[str, ...]:
    """Return deterministically ordered native target module paths.

    Returns
    -------
    tuple[str, ...]
        Import paths for native modules, sorted within each domain.
    """
    root = _native_root_dir()
    module_paths: list[str] = []
    for domain in _NATIVE_DOMAINS:
        domain_dir = root / domain
        if not domain_dir.is_dir():
            continue
        module_names: list[str] = []
        for path in domain_dir.iterdir():
            if path.is_file() and path.suffix == ".py":
                if path.name == "__init__.py":
                    continue
                module_names.append(path.stem)
                continue
            if path.is_dir() and (path / "__init__.py").is_file():
                module_names.append(path.name)
        module_paths.extend(
            f"{_NATIVE_PACKAGE_PREFIX}.{domain}.{name}" for name in sorted(module_names)
        )
    return tuple(module_paths)


@lru_cache(maxsize=1)
def load_native_modules() -> tuple[ModuleType, ...]:
    """Load all native target modules for driver composition.

    Returns
    -------
    tuple[ModuleType, ...]
        Tuple of imported native target modules.
    """
    return tuple(importlib.import_module(module_path) for module_path in native_module_paths())


__all__ = [
    "load_native_modules",
    "native_module_paths",
]
