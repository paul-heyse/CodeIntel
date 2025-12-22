"""Canonical build target specs derived from the Hamilton catalog cache.

This module defines the single source of truth for OutputTarget *specs* used by
the build system. Specs are loaded from the canonical target catalog which is
generated via Hamilton introspection and cached in metadata storage.
"""

from __future__ import annotations

from codeintel.build.catalogs.canonical import load_target_catalog
from codeintel.build.targets import OutputTarget


def load_target_specs() -> tuple[OutputTarget, ...]:
    """Load the canonical OutputTarget specs from the catalog cache.

    Returns
    -------
    tuple[OutputTarget, ...]
        Deterministically ordered OutputTarget specifications.
    """
    catalog = load_target_catalog()
    return tuple(catalog[name] for name in sorted(catalog))


__all__ = [
    "load_target_specs",
]
