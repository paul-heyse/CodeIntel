"""Unified plugin infrastructure for graphs and analytics.

This package provides a single plugin protocol used by both the graphs
and analytics subsystems, eliminating protocol duplication.

Modules
-------
- protocol: Unified plugin protocol and metadata types
- result: Unified plugin result types
- context: Unified plugin execution context
- registry: Unified plugin registry with dependency resolution
- executor: Unified plugin executor with retry, timeout, telemetry
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

__all__ = [
    "CapabilityKind",
    "ConfigProvider",
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
    "PluginProtocol",
    "PluginResourceHints",
    "PluginResult",
    "PluginScratch",
    "PluginSeverity",
    "PluginStage",
    "PluginStatus",
    "ResourceNotFoundError",
    "ResourceRegistry",
    "ValidationResult",
]
