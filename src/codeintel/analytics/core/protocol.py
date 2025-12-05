"""Analytics plugin protocol.

This module defines the protocol for analytics plugins, extending the
core unified plugin protocol with analytics-specific context requirements.

Architecture
------------
- `AnalyticsPluginProtocol` is a structural extension of `PluginProtocol`
  from `codeintel.core.plugins.protocol`.
- Analytics plugins use the extended `PluginExecutionContext` from
  `codeintel.analytics.core.context` which adds `scope` (AnalyticsScope).
- All metadata types (`PluginMetadata`, `PluginKind`, `PluginStage`, etc.)
  are inherited directly from the core unified types.
- This ensures type consistency across the codebase while allowing
  analytics-specific context requirements.

The analytics `PluginExecutionContext` extends the core context, so:
- Analytics plugins are structurally compatible with the core protocol
- Any analytics plugin can be used where a core protocol is expected
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.core.plugins import (
    CapabilityKind,
    InputSource,
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

# Note: PluginProtocol imported for re-export and documentation purposes.
# AnalyticsPluginProtocol is structurally compatible with it.
from codeintel.core.plugins.types.protocol import PluginProtocol
from codeintel.core.plugins.types.result import (
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)

if TYPE_CHECKING:
    from codeintel.analytics.core.context import PluginExecutionContext


@runtime_checkable
class AnalyticsPluginProtocol(Protocol):
    """Protocol for analytics plugins.

    Extend the core `PluginProtocol` with analytics-specific context.

    This protocol uses `codeintel.analytics.core.context.PluginExecutionContext`
    which extends the core context with analytics-specific fields like `scope`.

    Structural Compatibility
    ------------------------
    This protocol is structurally compatible with `PluginProtocol`:
    - Same `metadata` property returning `PluginMetadata`
    - Same `execute()` and `validate_inputs()` signatures
    - The analytics context extends the core context

    Any implementation satisfying this protocol also satisfies `PluginProtocol`.

    See Also
    --------
    codeintel.core.plugins.protocol.PluginProtocol
        The core unified plugin protocol.
    codeintel.analytics.core.context.PluginExecutionContext
        Analytics-specific execution context extending core context.
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Return the same core `PluginMetadata` type used across all domains.

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
            Analytics execution context (extends core context with scope).

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
            Analytics execution context to validate against.

        Returns
        -------
        ValidationResult
            Validation result with any errors.
        """
        ...


def is_analytics_plugin(plugin: object) -> bool:
    """Check if an object implements AnalyticsPluginProtocol.

    Parameters
    ----------
    plugin
        Object to check.

    Returns
    -------
    bool
        True if the object implements AnalyticsPluginProtocol.
    """
    return isinstance(plugin, AnalyticsPluginProtocol)


def is_core_compatible(plugin: AnalyticsPluginProtocol) -> bool:
    """Check if an analytics plugin is compatible with the core protocol.

    All analytics plugins are compatible with the core protocol since
    AnalyticsPluginProtocol is a structural extension of PluginProtocol.
    The analytics context extends the core context, so any analytics plugin
    can be used where a core plugin is expected.

    Parameters
    ----------
    plugin
        Analytics plugin to check.

    Returns
    -------
    bool
        Always True for valid analytics plugins.

    Notes
    -----
    This function always returns True because AnalyticsPluginProtocol is
    structurally compatible with PluginProtocol by design. It exists for
    documentation and explicit type checking purposes.
    """
    # Analytics plugins are always core-compatible due to structural typing
    # Check for required attributes directly instead of isinstance on Protocol
    return (
        hasattr(plugin, "metadata")
        and hasattr(plugin, "execute")
        and hasattr(plugin, "validate_inputs")
    )


# Re-export everything needed by analytics plugins
__all__ = [
    "AnalyticsPluginProtocol",
    "CapabilityKind",
    "InputSource",
    "PluginExecutionRecord",
    "PluginInputSpec",
    "PluginIsolation",
    "PluginKind",
    "PluginMetadata",
    "PluginOutputSpec",
    "PluginProtocol",
    "PluginResourceHints",
    "PluginResult",
    "PluginSeverity",
    "PluginStage",
    "PluginStatus",
    "ValidationResult",
    "is_analytics_plugin",
    "is_core_compatible",
]
