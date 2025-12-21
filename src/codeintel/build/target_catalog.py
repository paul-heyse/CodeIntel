"""Canonical build target specs derived from native Hamilton modules.

This module defines the single source of truth for OutputTarget *specs* used by
the build system. Specs are compiled from native Hamilton modules, which declare
`TARGET_SPECS` alongside the materialize nodes they implement.

Design notes
------------
- Dependencies are derived from the Hamilton DAG at runtime.
- Metadata (contracts, resources, execution policy, descriptions) lives next to
  native Hamilton implementations and is collected into a deterministic spec list.
"""

from __future__ import annotations

from codeintel.build.catalogs.canonical import load_target_catalog
from codeintel.build.target_specs import load_native_target_specs
from codeintel.build.targets import OutputTarget


def load_target_specs() -> tuple[OutputTarget, ...]:
    """Load the canonical OutputTarget specs from native Hamilton modules.

    Returns
    -------
    tuple[OutputTarget, ...]
        Deterministically ordered OutputTarget specifications.
    """
    try:
        catalog = load_target_catalog()
        return tuple(catalog[name] for name in sorted(catalog))
    except (KeyError, TypeError, ValueError, RuntimeError):
        return load_native_target_specs()


__all__ = [
    "load_target_specs",
]
