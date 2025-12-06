"""Graph plugin runtime infrastructure.

This package provides the execution infrastructure for graph plugins,
including planning, execution, and caching without any dependency
on the analytics subsystem.

Key Components
--------------
- GraphPluginExecutor: Executor using common infrastructure
- plan_graph_plugin_run: Creates an execution plan from plugin names
- GraphRunReport: Report of plugin execution outcomes
"""

from codeintel.core.execution.errors import PLUGIN_CATCHABLE_ERRORS, PluginFatalError
from codeintel.graphs.runtime.graph_executor import (
    GraphExecutorContext,
    GraphPluginExecutor,
    GraphRunReport,
)
from codeintel.graphs.runtime.manifest import (
    GraphPluginManifest,
    InputHashPayload,
    ManifestState,
    compute_input_hash,
    compute_options_hash,
)
from codeintel.graphs.runtime.planning import (
    GraphPlanContext,
    GraphPluginExecutionPlan,
    GraphPluginRunOptions,
    PluginExecutionSettings,
    plan_graph_plugin_run,
)

__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "GraphExecutorContext",
    "GraphPlanContext",
    "GraphPluginExecutionPlan",
    "GraphPluginExecutor",
    "GraphPluginManifest",
    "GraphPluginRunOptions",
    "GraphRunReport",
    "InputHashPayload",
    "ManifestState",
    "PluginExecutionSettings",
    "PluginFatalError",
    "compute_input_hash",
    "compute_options_hash",
    "plan_graph_plugin_run",
]
