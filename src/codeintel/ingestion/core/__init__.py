"""Core plugin infrastructure for ingestion pipelines.

This package provides the class-based plugin architecture for ingestion,
mirroring the analytics plugin pattern for consistency across the codebase.

Architecture
------------
- `BaseIngestPlugin`: Abstract base with common patterns
- `TableWriterIngestPlugin`: For plugins that write to database tables
- `ConfiguredIngestPlugin[TConfig]`: Auto-inject typed configuration
- `ToolDependentIngestPlugin`: For plugins requiring external tools

Example
-------
>>> @dataclass
... class MyPlugin(ConfiguredIngestPlugin[MyStepConfig], TableWriterIngestPlugin):
...     '''Compute my ingestion.'''
...
...     output_tables = ("core.my_table",)
...     config_type = MyStepConfig
...
...     def compute(self, ctx: IngestExecutionContext) -> dict[str, int]:
...         # Pure business logic only
...         return {"core.my_table": rows_written}
"""

from __future__ import annotations

from codeintel.ingestion.core.base import (
    BaseIngestPlugin,
    ConfiguredIngestPlugin,
    ConfiguredTableWriterPlugin,
    ResolvedConfig,
    TableWriterIngestPlugin,
    ToolDependentIngestPlugin,
    TrackerRequiringPlugin,
    ValidationResult,
)
from codeintel.ingestion.core.execution_context import (
    IngestExecutionContext,
    ResourceNotFoundError,
)
from codeintel.ingestion.core.traits import (
    IncrementalIngestPlugin,
    IsolatedPlugin,
    ProgressReportingPlugin,
    RetryablePlugin,
    ToolAwarePlugin,
    TrackerAwarePlugin,
    WithCaching,
    WithCleanup,
    WithDependencyData,
    WithIncrementalSupport,
    WithProgressReporting,
    WithRetries,
    WithRowCounts,
    WithToolDependencies,
    get_plugin_traits,
    is_incremental,
    is_isolated,
    is_retryable,
    is_tool_aware,
    is_tracker_aware,
)

__all__ = [
    "BaseIngestPlugin",
    "ConfiguredIngestPlugin",
    "ConfiguredTableWriterPlugin",
    "IncrementalIngestPlugin",
    "IngestExecutionContext",
    "IsolatedPlugin",
    "ProgressReportingPlugin",
    "ResolvedConfig",
    "ResourceNotFoundError",
    "RetryablePlugin",
    "TableWriterIngestPlugin",
    "ToolAwarePlugin",
    "ToolDependentIngestPlugin",
    "TrackerAwarePlugin",
    "TrackerRequiringPlugin",
    "ValidationResult",
    "WithCaching",
    "WithCleanup",
    "WithDependencyData",
    "WithIncrementalSupport",
    "WithProgressReporting",
    "WithRetries",
    "WithRowCounts",
    "WithToolDependencies",
    "get_plugin_traits",
    "is_incremental",
    "is_isolated",
    "is_retryable",
    "is_tool_aware",
    "is_tracker_aware",
]
