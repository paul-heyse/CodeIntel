"""Docstrings ingest plugin.

This module provides `DocstringsIngestPlugin` that extracts docstrings
and persists structured rows into core.docstrings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.errors import GatewayNotAvailableError
from codeintel.build.plugin import FactoryPlugin
from codeintel.build.plugins.ingestion.helpers import get_module_paths, paths_to_modules
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import DocstringsExtractStep

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.plugin import DiscoveryFactory, StorageFactory

log = logging.getLogger(__name__)


DOCSTRINGS_METADATA = CorePluginMetadata(
    name="ingest.docstrings",
    version="3.0.0",
    description="Extract docstrings and persist structured rows into core.docstrings.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="docstrings",
    provides=("core.docstrings",),
    requires=("core.modules",),
    produces_tables=("core.docstrings",),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
)


class DocstringsIngestPlugin(FactoryPlugin[DocstringsExtractStep]):
    """Extract docstrings and persist structured rows into core.docstrings.

    This plugin parses Python source files to extract docstrings from
    modules, classes, and functions, persisting structured information
    for documentation analysis.

    Outputs
    -------
    - core.docstrings: Structured docstring data
    """

    _core_metadata: ClassVar[CorePluginMetadata] = DOCSTRINGS_METADATA

    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[type[DocstringsExtractStep]] = DocstringsExtractStep

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute docstring extraction.

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
            context = "docstring extraction"
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
                log.warning("Docstring extraction error: %s", error)

        return TargetResult.succeeded(row_counts=result.table_counts or {})


__all__ = [
    "DOCSTRINGS_METADATA",
    "DocstringsIngestPlugin",
]
