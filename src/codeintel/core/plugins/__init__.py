"""Unified plugin infrastructure for graphs and analytics.

This package provides a single plugin protocol used by both the graphs
and analytics subsystems, eliminating protocol duplication.

Modules
-------
- protocol: Unified plugin protocol and metadata types
- result: Unified plugin result types
- context: Unified plugin execution context
- traits: Domain-agnostic plugin traits and mixins
- registry: Base plugin registry with dependency resolution
- functional: Base functional plugin for wrapping callables
- meta_options: Base plugin metadata options for decorators
- decorators: Plugin decorator factory functions
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
from codeintel.core.plugins.decorators import make_plugin_instance
from codeintel.core.plugins.functional import BaseFunctionalPlugin
from codeintel.core.plugins.meta_options import (
    BasePluginMetaOptions,
    BasePluginMetaOptionsInput,
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
from codeintel.core.plugins.registry import (
    BasePluginRegistry,
    PluginPlan,
    PluginSkip,
    RegistrablePlugin,
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
    "BaseFunctionalPlugin",
    "BasePluginMetaOptions",
    "BasePluginMetaOptionsInput",
    "BasePluginRegistry",
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
    "ValidationResult",
    "is_cache_aware",
    "is_isolated",
    "is_progress_reporting",
    "is_retryable",
    "make_plugin_instance",
]
