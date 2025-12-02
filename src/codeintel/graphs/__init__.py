"""Graph builders, plugins, and recipes for code graph construction and analysis.

This package provides the unified graph plugin architecture for building
and analyzing code graphs without dependency on the analytics subsystem.

Package Structure
-----------------
Core Infrastructure (graphs/core/):
- GraphPluginProtocol: Unified interface for all graph plugins
- GraphExecutionContext: Execution context providing storage and engine access
- GraphPluginRegistry: Central registry with dependency resolution
- graph_plugin: Decorator for defining graph plugins from functions

Runtime (graphs/runtime/):
- GraphPluginExecutor: Executes plugins with retry and timeout handling
- plan_graph_plugin_run: Creates an execution plan from plugin names
- run_graph_plugins: Executes a plan and returns a report

Recipes (graphs/recipes/):
- GraphRecipe: Declarative recipe definition
- RecipeExecutor: Executes recipes with stage orchestration
- Builtin recipes: full, incremental, metrics_only, validation_only

Plugins (graphs/plugins/):
- builders: Graph construction plugins (goid, callgraph, cfg_dfg, import_graph)
- metrics: Graph metric computation plugins
- validation: Graph validation plugin

Consolidated Domain Packages:
- callgraph/: Call graph edge collection, resolution, and persistence
- catalog/: Function catalog (spans, metadata, service) - unified module
- validation/: Graph validation checks, findings, and orchestration
- engine/: Graph engine protocol, NetworkX implementation, and views

Example
-------
```python
from codeintel.graphs.core import get_graph_registry, plan_graph_plugins
from codeintel.graphs.recipes import FULL_GRAPH_RECIPE, execute_graph_recipe

# Run a full graph construction and analysis pipeline
result = execute_graph_recipe(
    FULL_GRAPH_RECIPE,
    gateway=gateway,
    snapshot=snapshot,
)

# Or plan and run specific plugins
plan = plan_graph_plugins(["goid_builder", "callgraph_builder"])
```

Migration Notes
---------------
Several modules have been consolidated into domain-specific packages:

- Call graph modules (call_ast, call_cst, call_resolution, call_context,
  call_persist, import_resolver) -> ``codeintel.graphs.callgraph``

- Catalog modules (function_index, function_catalog, function_catalog_service)
  -> ``codeintel.graphs.catalog``

- Validation module (validation.py) -> ``codeintel.graphs.validation``

- Engine modules (engine.py, nx_views.py) -> ``codeintel.graphs.engine``

The original import paths remain functional via deprecation shims but will
emit warnings. Please update to the new canonical import paths.
"""

from __future__ import annotations

# Re-export key types from submodules for convenience
from codeintel.graphs.core import (
    GraphExecutionContext,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginResult,
    get_graph_registry,
    plan_graph_plugins,
    register_graph_plugin,
)
from codeintel.graphs.engine import GraphEngine, GraphKind, NxGraphEngine
from codeintel.graphs.recipes import (
    BUILDERS_ONLY_RECIPE,
    FULL_GRAPH_RECIPE,
    METRICS_ONLY_RECIPE,
    GraphRecipe,
    RecipeExecutionResult,
    execute_graph_recipe,
)

__all__ = [
    "BUILDERS_ONLY_RECIPE",
    "FULL_GRAPH_RECIPE",
    "METRICS_ONLY_RECIPE",
    "GraphEngine",
    "GraphExecutionContext",
    "GraphKind",
    "GraphPluginMetadata",
    "GraphPluginProtocol",
    "GraphPluginResult",
    "GraphRecipe",
    "NxGraphEngine",
    "RecipeExecutionResult",
    "execute_graph_recipe",
    "get_graph_registry",
    "plan_graph_plugins",
    "register_graph_plugin",
]
