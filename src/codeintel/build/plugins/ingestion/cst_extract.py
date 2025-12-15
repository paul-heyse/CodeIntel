"""CST extraction plugin.

This module provides `CstExtractPlugin` that parses CST via LibCST
and writes rows into core.cst_nodes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.errors import GatewayNotAvailableError
from codeintel.build.plugin import FactoryPlugin
from codeintel.build.plugins.ingestion.helpers import get_module_paths, paths_to_modules
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute import CstExtractStep

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.plugin import DiscoveryFactory, StorageFactory

log = logging.getLogger(__name__)


CST_EXTRACT_METADATA = CorePluginMetadata(
    name="ingest.cst_extract",
    version="3.0.0",
    description="Build concrete syntax trees and persist cst rows.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="cst",
    provides=("core.cst_nodes",),
    requires=("core.modules",),
    produces_tables=("core.cst_nodes",),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
)


class CstExtractPlugin(FactoryPlugin[CstExtractStep]):
    """Parse CST via LibCST and write rows into core.cst_nodes.

    This plugin parses Python source files using LibCST, extracting
    concrete syntax tree nodes for detailed analysis.

    Outputs
    -------
    - core.cst_nodes: CST node information
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CST_EXTRACT_METADATA

    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[type[CstExtractStep]] = CstExtractStep

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute CST extraction.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.

        Raises
        ------
        GatewayNotAvailableError
            If no storage gateway is available.
        """
        gateway = ctx.resources.gateway
        if gateway is None:
            context = "CST extraction"
            raise GatewayNotAvailableError(context)
        storage = self._storage_factory(gateway)
        discovery = self._discovery_factory(ctx.repo_root)

        paths = get_module_paths(ctx)
        modules = paths_to_modules(paths, ctx.repo_root)

        step = self._step_factory(storage, discovery)
        result = step.execute(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
        )

        if result.errors:
            for error in result.errors:
                log.warning("CST extraction error: %s", error)

        return TargetResult.succeeded(row_counts=result.table_counts or {})


__all__ = [
    "CST_EXTRACT_METADATA",
    "CstExtractPlugin",
]
