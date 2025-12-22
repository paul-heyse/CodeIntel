"""Canonical build target specs derived from the Hamilton catalog cache.

This module defines the single source of truth for OutputTarget *specs* used by
the build system. Specs are loaded from the canonical target catalog which is
generated via Hamilton introspection and cached in metadata storage.
"""

from __future__ import annotations

from functools import lru_cache

from codeintel.build.catalogs.canonical import load_target_catalog
from codeintel.build.targets import OutputTarget, TargetGraph


def load_target_specs() -> tuple[OutputTarget, ...]:
    """Load the canonical OutputTarget specs from the catalog cache.

    Returns
    -------
    tuple[OutputTarget, ...]
        Deterministically ordered OutputTarget specifications.
    """
    catalog = load_target_catalog()
    return tuple(catalog[name] for name in sorted(catalog))


@lru_cache(maxsize=1)
def target_graph_from_catalog() -> TargetGraph:
    """Build a TargetGraph from the canonical target catalog.

    Returns
    -------
    TargetGraph
        Graph built from canonical OutputTarget metadata.

    Raises
    ------
    ValueError
        If the catalog contains invalid or cyclic dependencies.
    """
    graph = TargetGraph()
    for target in load_target_specs():
        graph.register(target)
    errors = graph.validate()
    if errors:
        msg = "Invalid target catalog: " + "; ".join(errors)
        raise ValueError(msg)
    return graph


__all__ = [
    "load_target_specs",
    "target_graph_from_catalog",
]
