"""Unified plugin infrastructure for graphs and analytics.

This package provides a single plugin protocol used by both the graphs
and analytics subsystems, eliminating protocol duplication.

Subpackages
-----------
- types: Protocol, result, and report types
- execution: Context, executor, policy, and tracking
- registry: Base registry and plan types
- traits: Domain-agnostic plugin traits and mixins
- decorators: Functional plugins and meta options
"""

from __future__ import annotations

# Decorators
from codeintel.core.plugins.decorators.functional import BaseFunctionalPlugin
from codeintel.core.plugins.decorators.meta import (
    BasePluginMetaOptions,
    BasePluginMetaOptionsInput,
)
from codeintel.core.plugins.decorators.step import make_plugin_instance

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

# Traits
from codeintel.core.plugins.traits import (
    CacheAwareMixin,
    CacheAwarePlugin,
    IncrementalPlugin,
    IsolatedPlugin,
    ProgressReportingMixin,
    ProgressReportingPlugin,
    RetryableMixin,
    RetryablePlugin,
    ScratchContext,
    WithDependencyData,
    get_retry_policy,
    is_cache_aware,
    is_incremental,
    is_isolated,
    is_progress_reporting,
    is_retryable,
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
    # Decorators
    "BaseFunctionalPlugin",
    # Types - Result
    "BasePluginExecutionRecord",
    "BasePluginExecutor",
    "BasePluginMetaOptions",
    "BasePluginMetaOptionsInput",
    # Registry
    "BasePluginRegistry",
    "BasePluginResult",
    # Traits
    "CacheAwareMixin",
    "CacheAwarePlugin",
    # Types - Protocol
    "CapabilityKind",
    "CapabilityProvider",
    "ConfigProvider",
    "ExecutionStatus",
    "IncrementalPlugin",
    "InputSource",
    "IsolatedPlugin",
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
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "RegistrablePlugin",
    "ResourceNotFoundError",
    "ResourceRegistry",
    "RetryableMixin",
    "RetryablePlugin",
    "ScratchContext",
    "ValidationResult",
    "WithDependencyData",
    "build_provider_index",
    "build_provider_index_from_metadata",
    "complete_run_from_records",
    "get_retry_policy",
    "is_cache_aware",
    "is_incremental",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
    "make_plugin_instance",
    "record_plugin_steps",
    "topological_sort",
]
