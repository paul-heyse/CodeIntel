"""Unified plugin infrastructure for graphs and analytics.

This package provides a single plugin protocol used by both the graphs
and analytics subsystems, eliminating protocol duplication.

Subpackages
-----------
- types: Protocol, result, and report types
- execution: Context and manifest
- registry: Base registry and plan types
"""

from __future__ import annotations

from codeintel.core.plugins.execution.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
)
from codeintel.core.plugins.registry.base import (
    BasePluginRegistry,
    DefaultRegistryHooks,
    PluginPlan,
    PluginSkip,
    RegistrablePlugin,
    RegistryEntry,
    RegistryHooks,
)
from codeintel.core.plugins.registry.sorting import (
    CapabilityProvider,
    build_provider_index,
    build_provider_index_from_metadata,
    topological_sort,
)
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
    ValidationOutcome,
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
from codeintel.core.resources.registry import (
    ResourceNotFoundError,
    ResourceRegistry,
)

__all__ = [
    "BaseExecutionReport",
    "BasePluginExecutionRecord",
    "BasePluginRegistry",
    "BasePluginResult",
    "CapabilityKind",
    "CapabilityProvider",
    "ConfigProvider",
    "DefaultRegistryHooks",
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
    "RegistryEntry",
    "RegistryHooks",
    "ResourceNotFoundError",
    "ResourceRegistry",
    "ValidationOutcome",
    "build_provider_index",
    "build_provider_index_from_metadata",
    "topological_sort",
]
