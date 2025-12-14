"""AST extraction plugin.

This module provides `AstExtractPlugin` that parses Python AST
and persists rows + metrics into core.ast_* tables.
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
from codeintel.ingestion.compute import AstExtractStep

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.plugin import DiscoveryFactory, StorageFactory

log = logging.getLogger(__name__)


AST_EXTRACT_METADATA = CorePluginMetadata(
    name="ingest.ast_extract",
    version="3.0.0",
    description="Parse Python AST and persist rows + metrics into core.ast_* tables.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="ast",
    provides=("core.ast_nodes", "core.ast_metrics"),
    requires=("core.modules",),
    produces_tables=("core.ast_nodes", "core.ast_metrics"),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
)


class AstExtractPlugin(FactoryPlugin[AstExtractStep]):
    """Parse Python AST and persist rows + metrics.

    This plugin parses Python source files using the stdlib AST module,
    extracting node information and computing metrics.

    Outputs
    -------
    - core.ast_nodes: AST node information
    - core.ast_metrics: File-level AST metrics
    """

    _core_metadata: ClassVar[CorePluginMetadata] = AST_EXTRACT_METADATA

    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[AstExtractStep] = AstExtractStep  # type: ignore[assignment]

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute AST extraction.

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
            context = "AST extraction"
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
                log.warning("AST extraction error: %s", error)

        return TargetResult.succeeded(row_counts=result.table_counts or {})


__all__ = [
    "AST_EXTRACT_METADATA",
    "AstExtractPlugin",
]
