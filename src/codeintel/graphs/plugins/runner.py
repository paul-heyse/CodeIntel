"""Graph plugin execution service for pipeline integration.

This module provides a high-level interface for running graph plugins
from pipeline steps while properly integrating with the plugin architecture.

The runner handles:
- Context creation with proper resource injection
- Plugin execution with error propagation
- Resource cleanup and lifecycle management

Example
-------
```python
runner = GraphPluginRunner(gateway=gateway)
plugin = get_callgraph_plugin()
ctx = runner.build_context(cfg, catalog_provider=catalog)
runner.run_plugin(plugin, ctx)  # Raises GraphPluginError on failure
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.graphs.core.context import GraphExecutionContext, GraphRuntimeScratch
from codeintel.graphs.core.result import GraphPluginResult
from codeintel.graphs.resources.catalog import CatalogResource
from codeintel.graphs.resources.container import ResourceContainer
from codeintel.graphs.resources.storage import StorageResource

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.graphs.core.protocol import GraphPluginProtocol
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class GraphPluginError(Exception):
    """Raised when a graph plugin execution fails.

    Attributes
    ----------
    plugin_name
        Name of the plugin that failed.
    message
        Error message from the plugin.
    """

    def __init__(self, plugin_name: str, message: str) -> None:
        """Initialize the error.

        Parameters
        ----------
        plugin_name
            Name of the plugin that failed.
        message
            Error message describing the failure.
        """
        self.plugin_name = plugin_name
        self.message = message
        super().__init__(f"{plugin_name} failed: {message}")


@dataclass
class GraphPluginRunner:
    """Execute graph plugins with proper resource injection.

    This service provides the bridge between pipeline steps and the
    graph plugin architecture, handling context creation and error
    propagation.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    scratch
        Optional shared scratch space for plugin communication.
    """

    gateway: StorageGateway
    scratch: GraphRuntimeScratch | None = None

    def run_plugin(
        self,
        plugin: GraphPluginProtocol,
        ctx: GraphExecutionContext,
        *,
        raise_on_failure: bool = True,
    ) -> GraphPluginResult:
        """Execute a single graph plugin.

        Parameters
        ----------
        plugin
            The plugin to execute.
        ctx
            Pre-configured execution context.
        raise_on_failure
            If True, raise GraphPluginError on failure.

        Returns
        -------
        GraphPluginResult
            Plugin execution result.

        Raises
        ------
        GraphPluginError
            If plugin fails and raise_on_failure is True.
        """
        ctx = self._with_scratch(ctx, self.scratch)

        plugin_name = plugin.metadata.name
        log.debug("Running graph plugin: %s", plugin_name)

        result = plugin.execute(ctx)

        if not result.success and raise_on_failure:
            error_msg = result.error or "Unknown error"
            raise GraphPluginError(plugin_name, error_msg)

        if result.success:
            log.debug("Plugin %s completed successfully", plugin_name)
        else:
            log.warning("Plugin %s failed: %s", plugin_name, result.error)

        return result

    @staticmethod
    def _with_scratch(
        ctx: GraphExecutionContext,
        scratch: GraphRuntimeScratch | None,
    ) -> GraphExecutionContext:
        """Ensure the context has scratch space configured.

        Since GraphExecutionContext.scratch has a default factory, it always
        has scratch space. If the runner has shared scratch data, we copy
        entries into the context's scratch.

        Returns
        -------
        GraphExecutionContext
            Context ready for plugin execution.
        """
        if scratch is not None:
            # Copy shared scratch entries into context's scratch space
            # keys() returns a tuple of keys, iterate to copy each entry
            scratch_keys = scratch.keys()
            for key in scratch_keys:
                value = scratch.consume(key)
                if value is not None:
                    ctx.scratch.declare(key, value)
        return ctx

    def build_context(
        self,
        snapshot: SnapshotRef,
        *,
        catalog_provider: FunctionCatalogProvider | None = None,
    ) -> GraphExecutionContext:
        """Build a properly configured execution context.

        Parameters
        ----------
        snapshot
            Snapshot reference with repo, commit, and repo_root.
        catalog_provider
            Optional function catalog for enrichment.

        Returns
        -------
        GraphExecutionContext
            Ready-to-use execution context with resources registered.
        """
        container = ResourceContainer()
        container.register(StorageResource(self.gateway, snapshot.repo_root))

        # Register catalog if provided - call .catalog() to get the FunctionCatalog
        if catalog_provider is not None:
            container.register(CatalogResource(catalog_provider.catalog()))

        return GraphExecutionContext(
            snapshot=snapshot,
            resources=container,
            scratch=self.scratch or GraphRuntimeScratch(),
        )


__all__ = [
    "GraphPluginError",
    "GraphPluginRunner",
]
