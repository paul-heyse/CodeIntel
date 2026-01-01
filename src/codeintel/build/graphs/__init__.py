"""Graph builders, plugins, and infrastructure for code graph construction and analysis.

This package provides graph construction and analysis capabilities integrated
with the Hamilton build system.

Package Structure
-----------------
Plugins (graphs/plugins/):
- builders: Graph construction plugins (goid, callgraph, cfg_dfg, import_graph)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

Hexagonal Architecture (Ports, Compute, Resources):
- ports/: Protocol interfaces abstracting I/O (StoragePort, CatalogPort, etc.)
- compute/: Pure stateless computation functions (no I/O)
- resources/: DI container and resource providers

Consolidated Domain Packages:
- core.catalog: Function catalog (spans, metadata, service)
- validation/: Graph validation checks, findings, and orchestration
- engine/: Graph engine protocol, NetworkX implementation, and views

Callgraph logic is in compute/callgraph/ (pure functions and persistence utilities).

Example
-------
```python
from codeintel.build.graphs.compute.metrics import centrality
from codeintel.runtime.compose import compose_runtime


runtime = compose_runtime(env=env).bundle
catalog = runtime.catalog
graph_targets = [t for t in catalog.all_targets if t.module == "graphs"]


pagerank = centrality.compute_pagerank(call_graph)
```

Architecture Notes
------------------
The graphs package uses hexagonal architecture:

- Plugins are the orchestration layer, composing resources and compute functions
- Resources provide injectable access to I/O (storage, engine, catalog)
- Compute functions are pure and stateless, taking data and returning data
- Ports define protocol interfaces for abstraction

All builder modules have been consolidated into their corresponding plugins
under plugins/builders/. Pure computation logic is in compute/.

Graph/target integration is driven by the build target metadata service in
`codeintel.build.target_metadata`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.graphs import compute, ports, resources
    from codeintel.build.graphs.engine import GraphEngine, GraphKind, NxGraphEngine
    from codeintel.build.graphs.resources import ResourceProvider
    from codeintel.core.catalog import CatalogService, FunctionSpan
    from codeintel.core.resources import ResourceRegistry

__all__ = [
    "CatalogService",
    "FunctionSpan",
    "GraphEngine",
    "GraphKind",
    "NxGraphEngine",
    "ResourceProvider",
    "ResourceRegistry",
    "compute",
    "ports",
    "resources",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "CatalogService": ("codeintel.core.catalog", "CatalogService"),
    "FunctionSpan": ("codeintel.core.catalog", "FunctionSpan"),
    "GraphEngine": ("codeintel.build.graphs.engine", "GraphEngine"),
    "GraphKind": ("codeintel.build.graphs.engine", "GraphKind"),
    "NxGraphEngine": ("codeintel.build.graphs.engine", "NxGraphEngine"),
    "ResourceProvider": ("codeintel.build.graphs.resources", "ResourceProvider"),
    "ResourceRegistry": ("codeintel.core.resources", "ResourceRegistry"),
}


def __getattr__(name: str) -> object:
    """Lazily resolve package exports without importing heavy submodules at import time.

    Parameters
    ----------
    name
        Attribute name requested by the caller.

    Returns
    -------
    object
        The resolved module attribute.

    Raises
    ------
    AttributeError
        If the attribute is not supported by this package.
    """
    if name in {"compute", "ports", "resources"}:
        return importlib.import_module(f"{__name__}.{name}")
    lazy = _LAZY_ATTRS.get(name)
    if lazy is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_path, attr_name = lazy
    return getattr(importlib.import_module(module_path), attr_name)


def __dir__() -> list[str]:
    """Return module attributes including lazy exports.

    Returns
    -------
    list[str]
        Attribute names visible on the module.
    """
    return sorted({*globals().keys(), *__all__})
