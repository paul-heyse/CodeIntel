"""Graph builders, plugins, and infrastructure for code graph construction and analysis.

This package provides the unified graph plugin architecture for building
and analyzing code graphs without dependency on the analytics subsystem.

Package Structure
-----------------
Core Infrastructure (graphs/core/):
- GraphPluginProtocol: Unified interface for all graph plugins
- GraphPluginExecutionContext: Execution context providing storage and engine access
- GraphPluginRegistry: Central registry with dependency resolution
- graph_plugin: Decorator for defining graph plugins from functions

Runtime (graphs/runtime/):
- GraphPluginExecutor: Executes plugins with retry and timeout handling
- plan_graph_plugin_run: Creates an execution plan from plugin names
- run_graph_plugins: Executes a plan and returns a report

Plugins (graphs/plugins/):
- builders: Graph construction plugins (goid, callgraph, cfg_dfg, import_graph)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

Hexagonal Architecture (Ports, Adapters, Compute, Resources):
- ports/: Protocol interfaces abstracting I/O (StoragePort, ParsingPort, etc.)
- adapters/: Concrete implementations of ports (DuckDB, LibCST, etc.)
- compute/: Pure stateless computation functions (no I/O)
- resources/: DI container and resource providers

Consolidated Domain Packages:
- catalog/: Function catalog (spans, metadata, service) - unified module
- validation/: Graph validation checks, findings, and orchestration
- engine/: Graph engine protocol, NetworkX implementation, and views

Callgraph logic is in compute/callgraph.py (pure functions) and
adapters/callgraph_persistence.py (persistence).

Example
-------
```python
from codeintel.graphs.core import get_graph_registry, plan_graph_plugins

# Plan and run specific plugins
plan = plan_graph_plugins(["goid_builder", "callgraph_builder"])

# Using hexagonal architecture
from codeintel.core.resources import ResourceRegistry
from codeintel.graphs.resources import StorageResource
from codeintel.graphs.compute.metrics import centrality

resources = ResourceRegistry()
resources.register_provider(StorageResource(gateway, repo_root))

# Pure computation with no I/O
pagerank = centrality.compute_pagerank(call_graph)
```

Architecture Notes
------------------
The graphs package uses hexagonal architecture:

- Plugins are the orchestration layer, composing resources and compute functions
- Resources provide injectable access to I/O (storage, engine, catalog)
- Compute functions are pure and stateless, taking data and returning data
- Ports define protocol interfaces, adapters provide concrete implementations

All builder modules have been consolidated into their corresponding plugins
under plugins/builders/. Pure computation logic is in compute/, persistence
logic is in adapters/.
"""

from __future__ import annotations

# Hexagonal architecture exports
from codeintel.core.resources import ResourceRegistry

# Subpackage re-exports for convenient access
from codeintel.graphs import adapters, compute, ports, resources

# Re-export key types from submodules for convenience
from codeintel.graphs.core import (
    GraphPluginExecutionContext,
    GraphPluginMetadata,
    GraphPluginProtocol,
    PluginResult,
    get_graph_registry,
    plan_graph_plugins,
    register_graph_plugin,
)
from codeintel.graphs.engine import GraphEngine, GraphKind, NxGraphEngine
from codeintel.graphs.resources import ResourceProvider

__all__ = [
    "GraphEngine",
    "GraphKind",
    "GraphPluginExecutionContext",
    "GraphPluginMetadata",
    "GraphPluginProtocol",
    "NxGraphEngine",
    "PluginResult",
    "ResourceProvider",
    "ResourceRegistry",
    "adapters",
    "compute",
    "get_graph_registry",
    "plan_graph_plugins",
    "ports",
    "register_graph_plugin",
    "resources",
]
