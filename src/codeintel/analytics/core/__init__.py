"""Core analytics plugin infrastructure.

This module provides the unified plugin protocol, registry, and execution context
for all analytics plugins. It implements a single, extensible architecture
supporting trait-based capabilities and composable recipes.

Key Components
--------------
- Base Classes: `BasePlugin`, `TableWriterPlugin`, `ConfigBoundPlugin`, etc.
- Builders: `PluginSpec`, `ResourceHints`, `OutputSpec` for fluent metadata construction
- Protocol: `AnalyticsPluginProtocol` defines the interface all plugins implement
- Registry: `PluginRegistry` for plugin discovery and dependency resolution
- Executor: `PluginExecutor` handles execution with error handling and telemetry
- Traits: Mixins for optional capabilities like caching, contracts, retries
"""

from __future__ import annotations

from codeintel.analytics.core.base import (
    AnalyticsContextRequiringPlugin,
    BasePlugin,
    CatalogRequiringPlugin,
    ConfigBoundPlugin,
    ConfiguredGraphMetricsPlugin,
    ConfiguredTableWriterPlugin,
    GraphMetricsPlugin,
    GraphRuntimeRequiringPlugin,
    TableWriterPlugin,
    capabilities_from_tables,
)
from codeintel.analytics.core.builders import (
    OutputSpec,
    OutputSpecBuilder,
    PluginSpec,
    PluginSpecBuilder,
    ResourceHints,
    ResourceHintsBuilder,
)
from codeintel.analytics.core.config_registry import (
    AnalyticsStepConfigBase,
    BaseStepConfig,
    ConfigPluginMapping,
    ConfigRegistry,
    get_config_registry,
    register_config,
)
from codeintel.analytics.core.contracts import (
    ColumnConstraint,
    ContractCheckerFn,
    ContractValidationResult,
    ContractValidator,
    ContractViolation,
    OutputContractSpec,
    PluginOutputContract,
    build_plugin_output_contracts,
    create_contract_checker,
    validate_plugin_outputs,
)
from codeintel.analytics.core.execution_context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.analytics.core.executor import (
    ExecutionPolicy,
    ExecutionReport,
    PluginExecutor,
    execute_plugin_plan,
)
from codeintel.analytics.core.plugin_protocol import (
    AnalyticsPluginProtocol,
    CapabilityKind,
    InputSource,
    PluginCapability,
    PluginExecutionRecord,
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    PluginSeverity,
    PluginStage,
    ValidationResult,
)
from codeintel.analytics.core.registry import (
    FunctionalPlugin,
    PluginPlan,
    PluginRegistry,
    PluginSkip,
    get_registry,
    plugin,
    register_plugin,
)
from codeintel.analytics.core.traits import (
    AnalyticsContextAwarePlugin,
    CacheAwareMixin,
    CacheAwarePlugin,
    CatalogAwarePlugin,
    ContractValidatedPlugin,
    GraphAwareMixin,
    GraphAwarePlugin,
    IncrementalPlugin,
    IsolatedPlugin,
    ProgressReportingPlugin,
    RetryableMixin,
    RetryablePlugin,
    ScopeAwareMixin,
    ScopeAwarePlugin,
    WithCaching,
    WithCleanup,
    WithContractValidation,
    WithDependencyData,
    WithProgressReporting,
    WithRowCounts,
    get_plugin_traits,
    is_contract_validated,
    is_graph_aware,
    is_incremental,
    is_isolated,
    is_scope_aware,
)

__all__ = [
    # Legacy exports (kept for compatibility)
    "AnalyticsContextAwarePlugin",
    # Base classes
    "AnalyticsContextRequiringPlugin",
    "AnalyticsPluginProtocol",
    "AnalyticsStepConfigBase",
    "BasePlugin",
    "BaseStepConfig",
    "CacheAwareMixin",
    "CacheAwarePlugin",
    "CapabilityKind",
    "CatalogAwarePlugin",
    "CatalogRequiringPlugin",
    "ColumnConstraint",
    "ConfigBoundPlugin",
    "ConfigPluginMapping",
    "ConfigProvider",
    "ConfigRegistry",
    "ConfiguredGraphMetricsPlugin",
    "ConfiguredTableWriterPlugin",
    "ContractCheckerFn",
    "ContractValidatedPlugin",
    "ContractValidationResult",
    "ContractValidator",
    "ContractViolation",
    "ExecutionPolicy",
    "ExecutionReport",
    "FunctionalPlugin",
    "GraphAwareMixin",
    "GraphAwarePlugin",
    "GraphMetricsPlugin",
    "GraphRuntimeRequiringPlugin",
    "IncrementalPlugin",
    "InputSource",
    "IsolatedPlugin",
    "OutputContractSpec",
    # Builders
    "OutputSpec",
    "OutputSpecBuilder",
    "PluginCapability",
    "PluginExecutionContext",
    "PluginExecutionContextBuilder",
    "PluginExecutionRecord",
    "PluginExecutor",
    "PluginInputSpec",
    "PluginMetadata",
    "PluginOutputContract",
    "PluginOutputSpec",
    "PluginPlan",
    "PluginRegistry",
    "PluginResourceHints",
    "PluginResult",
    "PluginScratch",
    "PluginSeverity",
    "PluginSkip",
    "PluginSpec",
    "PluginSpecBuilder",
    "PluginStage",
    "ProgressReportingPlugin",
    "ResourceHints",
    "ResourceHintsBuilder",
    "RetryableMixin",
    "RetryablePlugin",
    "ScopeAwareMixin",
    "ScopeAwarePlugin",
    "TableWriterPlugin",
    "ValidationResult",
    # Composition mixins
    "WithCaching",
    "WithCleanup",
    "WithContractValidation",
    "WithDependencyData",
    "WithProgressReporting",
    "WithRowCounts",
    "build_plugin_output_contracts",
    "capabilities_from_tables",
    "create_contract_checker",
    "execute_plugin_plan",
    "get_config_registry",
    "get_plugin_traits",
    "get_registry",
    "is_contract_validated",
    "is_graph_aware",
    "is_incremental",
    "is_isolated",
    "is_scope_aware",
    "plugin",
    "register_config",
    "register_plugin",
    "validate_plugin_outputs",
]
