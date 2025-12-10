"""Plugin adapter for operations.

Provides capability-gated operation execution for plugins,
enabling secure sandboxing of plugin code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from codeintel.operations.errors.factory import fail_capability_denied, fail_operation_not_found
from codeintel.operations.registry import OperationRegistry, get_default_registry
from codeintel.operations.result import Result

LOG = logging.getLogger(__name__)


@dataclass
class PluginAdapter:
    """Adapt operations for plugin invocation with capability gating.

    Plugins use this adapter to invoke operations, with capabilities
    checked before execution.

    Parameters
    ----------
    registry
        Operation registry (defaults to global).
    granted_capabilities
        Capabilities granted to this plugin.
    plugin_name
        Name of the plugin for logging/errors.
    """

    registry: OperationRegistry = field(default_factory=get_default_registry)
    granted_capabilities: frozenset[str] = field(default_factory=frozenset)
    plugin_name: str = "unknown"

    def invoke(self, operation_id: str, params: object) -> Result[object]:
        """Invoke an operation with capability checking.

        Parameters
        ----------
        operation_id
            ID of the operation to invoke.
        params
            Operation parameters.

        Returns
        -------
        Result[object]
            Operation result (success or failure).
        """
        _ = params  # Will be used when full implementation is added

        # Look up operation
        spec = self.registry.get(operation_id)
        if spec is None:
            LOG.warning(
                "Plugin %s attempted to invoke unknown operation: %s",
                self.plugin_name,
                operation_id,
            )
            return fail_operation_not_found(operation_id)

        # Check capabilities
        missing = spec.capabilities - self.granted_capabilities
        if missing:
            LOG.warning(
                "Plugin %s denied capability %s for operation %s",
                self.plugin_name,
                missing,
                operation_id,
            )
            return fail_capability_denied(next(iter(missing)), operation_id)

        # Placeholder - full implementation will execute the operation
        return Result.ok({"status": "placeholder"})

    def can_invoke(self, operation_id: str) -> bool:
        """Check if the plugin can invoke an operation.

        Parameters
        ----------
        operation_id
            ID of the operation to check.

        Returns
        -------
        bool
            True if the operation exists and plugin has capabilities.
        """
        spec = self.registry.get(operation_id)
        if spec is None:
            return False
        return spec.capabilities <= self.granted_capabilities

    def list_available(self) -> list[str]:
        """List operations available to this plugin.

        Returns
        -------
        list[str]
            Operation IDs the plugin can invoke.
        """
        return [
            spec.operation_id
            for spec in self.registry.list_operations()
            if spec.capabilities <= self.granted_capabilities
        ]


def create_plugin_adapter(
    plugin_name: str,
    capabilities: frozenset[str],
) -> PluginAdapter:
    """Create a plugin adapter with specific capabilities.

    Parameters
    ----------
    plugin_name
        Name of the plugin.
    capabilities
        Capabilities to grant.

    Returns
    -------
    PluginAdapter
        Configured adapter.
    """
    return PluginAdapter(
        granted_capabilities=capabilities,
        plugin_name=plugin_name,
    )


__all__ = [
    "PluginAdapter",
    "create_plugin_adapter",
]
