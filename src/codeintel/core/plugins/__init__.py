"""Unified plugin infrastructure for graphs and analytics.

This package provides a single plugin protocol used by both the graphs
and analytics subsystems, eliminating protocol duplication.

Modules
-------
- protocol: Unified plugin protocol and metadata types
- result: Unified plugin result types
- context: Unified plugin execution context
- traits: Domain-agnostic plugin traits and mixins
"""

from __future__ import annotations

from codeintel.core.plugins.context import (
    ConfigProvider,
    PluginExecutionContext,
    PluginExecutionContextBuilder,
    PluginScratch,
    ResourceNotFoundError,
    ResourceRegistry,
)
from codeintel.core.plugins.protocol import (
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
from codeintel.core.plugins.result import (
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)
from codeintel.core.plugins.traits import (
    CacheAwareMixin,
    CacheAwarePlugin,
    IsolatedPlugin,
    ProgressReportingMixin,
    ProgressReportingPlugin,
    RetryableMixin,
    RetryablePlugin,
    is_cache_aware,
    is_isolated,
    is_progress_reporting,
    is_retryable,
)

__all__ = [
    "CacheAwareMixin",
    "CacheAwarePlugin",
    "CapabilityKind",
    "ConfigProvider",
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
    "PluginProtocol",
    "PluginResourceHints",
    "PluginResult",
    "PluginScratch",
    "PluginSeverity",
    "PluginStage",
    "PluginStatus",
    "ProgressReportingMixin",
    "ProgressReportingPlugin",
    "ResourceNotFoundError",
    "ResourceRegistry",
    "RetryableMixin",
    "RetryablePlugin",
    "ValidationResult",
    "is_cache_aware",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
]
