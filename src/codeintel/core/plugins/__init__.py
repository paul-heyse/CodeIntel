"""Unified plugin infrastructure for graphs and analytics.

This package provides a single plugin protocol used by both the graphs
and analytics subsystems, eliminating protocol duplication.

Subpackages
-----------
- types: Protocol, result, and report types
- execution: Context, executor, policy, and tracking
- registry: Base registry and plan types
"""

from __future__ import annotations

# Execution
from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.plugins.execution.executor import BasePluginExecutor
from codeintel.core.plugins.execution.executor_context import BaseExecutorContext
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.execution.tracking import (
    complete_run_from_records,
    record_plugin_steps,
)

# Registry
from codeintel.core.plugins.registry.base import (
    BasePluginRegistry,
    PluginPlan,
    PluginSkip,
    RegistrablePlugin,
)
from codeintel.core.plugins.registry.sorting import (
    CapabilityProvider,
    build_provider_index,
    build_provider_index_from_metadata,
    topological_sort,
)

# Types
from codeintel.core.plugins.types.protocol import (
    CapabilityKind,
    InputSource,
    PluginCapability,
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
    PluginProtocol,
    PluginResourceHints,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.core.plugins.types.report import (
    BaseExecutionReport,
    ExecutionStatus,
)
from codeintel.core.plugins.types.result import (
    BasePluginExecutionRecord,
    BasePluginResult,
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)

# Re-export from resources (for backwards compatibility)
from codeintel.core.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)

__all__ = [
    # Execution
    "BaseExecutionPolicy",
    # Types - Report
    "BaseExecutionReport",
    "BaseExecutorContext",
    # Types - Result
    "BasePluginExecutionRecord",
    "BasePluginExecutor",
    # Registry
    "BasePluginRegistry",
    "BasePluginResult",
    # Types - Protocol
    "CapabilityKind",
    "CapabilityProvider",
    "ConfigProvider",
    "ExecutionStatus",
    "InputSource",
    "PluginCapability",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginExecutionRecord",
    "PluginInputSpec",
    "PluginIsolation",
    "PluginKind",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginPlan",
    "PluginProtocol",
    "PluginResourceHints",
    "PluginResult",
    "PluginScratch",
    "PluginSeverity",
    "PluginSkip",
    "PluginStage",
    "PluginStatus",
    "RegistrablePlugin",
    "ResourceNotFoundError",
    "ResourceRegistry",
    "ValidationResult",
    "build_provider_index",
    "build_provider_index_from_metadata",
    "complete_run_from_records",
    "record_plugin_steps",
    "topological_sort",
]
