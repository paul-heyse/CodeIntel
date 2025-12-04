"""Analytics plugin protocol.

This module defines the protocol for analytics plugins, using the
analytics-specific execution context.

The graph plugin system uses a separate protocol defined in
codeintel.core.plugins, which will be the unified protocol once
migration is complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.core.plugins import (
    CapabilityKind,
    InputSource,
    PluginCapability,
    PluginInputSpec,
    PluginIsolation,
    PluginKind,
    PluginMetadata,
    PluginOutputSpec,
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

if TYPE_CHECKING:
    from codeintel.analytics.core.execution_context import PluginExecutionContext


@runtime_checkable
class AnalyticsPluginProtocol(Protocol):
    """Protocol for analytics plugins.

    Analytics plugins use the analytics-specific PluginExecutionContext
    which provides access to analytics-specific resources and configuration.
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Metadata describing the plugin.
        """
        ...

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin with the given context.

        Parameters
        ----------
        ctx
            Execution context providing access to storage, config, and runtime.

        Returns
        -------
        PluginResult
            Result of the plugin execution.
        """
        ...

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context to validate against.

        Returns
        -------
        ValidationResult
            Validation result with any errors.
        """
        ...


# Alias for backward compatibility - some code uses PluginProtocol
PluginProtocol = AnalyticsPluginProtocol


# Re-export everything needed by analytics plugins
__all__ = [
    # Analytics protocol
    "AnalyticsPluginProtocol",
    "PluginProtocol",
    # Canonical unified types (from core.plugins)
    "CapabilityKind",
    "InputSource",
    "PluginCapability",
    "PluginExecutionRecord",
    "PluginInputSpec",
    "PluginIsolation",
    "PluginKind",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginResourceHints",
    "PluginResult",
    "PluginSeverity",
    "PluginStage",
    "PluginStatus",
    "ValidationResult",
]
