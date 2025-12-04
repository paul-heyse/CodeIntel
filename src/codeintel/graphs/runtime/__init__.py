"""Graph plugin runtime infrastructure.

This package provides the execution infrastructure for graph plugins,
including planning, execution, caching, and telemetry without any
dependency on the analytics subsystem.

Key Components
--------------
- GraphPluginExecutor: Executes plugins with retry and timeout handling
- plan_graph_plugin_run: Creates an execution plan from plugin names
- run_graph_plugins: Executes a plan and returns a report
"""

from codeintel.core.runtime.errors import PLUGIN_CATCHABLE_ERRORS, PluginFatalError
from codeintel.graphs.runtime.executor import (
    GraphExecutorContext,
    GraphRunReport,
    run_graph_plugin_batch,
    run_graph_plugins,
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
from codeintel.graphs.runtime.telemetry import (
    GraphPluginSpan,
    GraphRuntimeTelemetry,
    get_graph_telemetry,
)

__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "GraphExecutorContext",
    "GraphPlanContext",
    "GraphPluginExecutionPlan",
    "GraphPluginManifest",
    "GraphPluginRunOptions",
    "GraphPluginSpan",
    "GraphRunReport",
    "GraphRuntimeTelemetry",
    "InputHashPayload",
    "ManifestState",
    "PluginExecutionSettings",
    "PluginFatalError",
    "compute_input_hash",
    "compute_options_hash",
    "get_graph_telemetry",
    "plan_graph_plugin_run",
    "run_graph_plugin_batch",
    "run_graph_plugins",
]
