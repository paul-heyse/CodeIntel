"""Ingestion plugin runtime infrastructure.

This package provides the execution infrastructure for ingestion plugins,
including planning, execution, and telemetry. Analogous to graphs/runtime/
for structural alignment.

Key Components
--------------
- IngestExecutorConfig: Configuration for plugin execution
- IngestPluginExecutionRecord: Record of individual plugin execution (extends BasePluginExecutionRecord)
- execute_plugin_batch: Execute multiple plugins with optional parallelism
- plan_ingest_plugins: Create an execution plan from plugin names
- IngestRuntimeTelemetry: Telemetry integration with OpenTelemetry

Example
-------
```python
from codeintel.ingestion.runtime import (
    IngestExecutorConfig,
    execute_plugin_batch,
)
from codeintel.ingestion.plugins.registry import get_ingest_registry

# Get plugins
registry = get_ingest_registry()
plugins = [registry.get("ast_extract"), registry.get("cst_extract")]

# Execute with configuration
config = IngestExecutorConfig(run_id="run-123", enable_parallel=True)
report = execute_plugin_batch(plugins, context, config=config, parallel=True)
print(f"Executed {len(report.records)} plugins, {report.total_rows_written} rows")
```
"""

from codeintel.ingestion.runtime.executor import (
    PLUGIN_CATCHABLE_ERRORS,
    IngestExecutorConfig,
    IngestPluginExecutionRecord,
    IngestRunReport,
    PluginExecutionSettings,
    PluginFatalError,
    PluginSeverity,
    execute_plugin,
    execute_plugin_batch,
    execute_plugin_with_timeout,
)
from codeintel.ingestion.runtime.planning import (
    IngestPlanContext,
    PlanOptions,
    PluginExecutionPlan,
    plan_ingest_plugins,
    resolve_plugin_order,
)
from codeintel.ingestion.runtime.telemetry import (
    IngestPluginSpan,
    IngestRuntimeTelemetry,
    OtelIngestRunSink,
    get_ingest_telemetry,
)

__all__ = [
    "PLUGIN_CATCHABLE_ERRORS",
    "IngestExecutorConfig",
    "IngestPlanContext",
    "IngestPluginExecutionRecord",
    "IngestPluginSpan",
    "IngestRunReport",
    "IngestRuntimeTelemetry",
    "OtelIngestRunSink",
    "PlanOptions",
    "PluginExecutionPlan",
    "PluginExecutionSettings",
    "PluginFatalError",
    "PluginSeverity",
    "execute_plugin",
    "execute_plugin_batch",
    "execute_plugin_with_timeout",
    "get_ingest_telemetry",
    "plan_ingest_plugins",
    "resolve_plugin_order",
]
