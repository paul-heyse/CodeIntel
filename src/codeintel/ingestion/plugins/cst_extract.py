"""CST extraction plugin.

This module provides `CstExtractPlugin` that parses CST via LibCST
and writes rows into core.cst_nodes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute import CstExtractStep
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
from codeintel.ingestion.ports.storage import IngestStoragePort
from codeintel.storage.ibis_types import filter_by, ibis_bool

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage
    from codeintel.storage.gateway import StorageGateway
else:
    StorageGateway = object

log = logging.getLogger(__name__)

StorageFactory = Callable[[StorageGateway], IngestStoragePort]
DiscoveryFactory = Callable[[Path], ModuleDiscoveryPort]
StepFactory = Callable[[IngestStoragePort, ModuleDiscoveryPort], CstExtractStep]


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


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "cst"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


def _paths_to_modules(paths: list[str], repo_root: Path) -> list[ModuleRecord]:
    """Convert string paths to ModuleRecord objects.

    Returns
    -------
    list[ModuleRecord]
        Module records with metadata.
    """
    total = len(paths)
    return [
        ModuleRecord(
            rel_path=path,
            module_name=path.replace("/", ".").removesuffix(".py"),
            file_path=repo_root / path,
            index=i + 1,
            total=total,
        )
        for i, path in enumerate(paths)
    ]


def _get_module_paths(ctx: TargetExecutionContext) -> list[str]:
    """Get module paths from context resources or database.

    Returns
    -------
    list[str]
        List of relative module paths.
    """
    if ctx.resources.modules:
        return list(ctx.resources.modules)
    try:
        table = ctx.gateway.ibis.table("core.modules")
        df = (
            filter_by(
                table,
                ibis_bool(table.repo == ctx.repo),
                ibis_bool(table.commit == ctx.commit),
            )
            .select("path")
            .execute()
        )
        return [str(path) for path in df["path"].tolist()]
    except (RuntimeError, OSError):
        return []


class CstExtractPlugin(TargetPlugin):
    """Parse CST via LibCST and write rows into core.cst_nodes.

    This plugin parses Python source files using LibCST, extracting
    concrete syntax tree nodes for detailed analysis.

    Outputs
    -------
    - core.cst_nodes: CST node information
    """

    plugin_name: ClassVar[str] = "cst_extract"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Parse CST via LibCST and write rows into core.cst_nodes."
    _core_metadata: ClassVar[CorePluginMetadata] = CST_EXTRACT_METADATA

    # Class-level defaults for adapter and step factories
    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[StepFactory] = CstExtractStep

    # Instance attributes (set in __init__)
    _storage_factory: StorageFactory
    _discovery_factory: DiscoveryFactory
    _step_factory: StepFactory

    def __init__(
        self,
        *,
        storage_adapter_factory: StorageFactory | None = None,
        discovery_adapter_factory: DiscoveryFactory | None = None,
        step_factory: StepFactory | None = None,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        self._storage_factory = storage_adapter_factory or type(self).default_storage_factory
        self._discovery_factory = discovery_adapter_factory or type(self).default_discovery_factory
        self._step_factory = step_factory or type(self).default_step_factory
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata definition."""
        return self._core_metadata

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
        ValueError
            If no storage gateway is available.
        """
        # Create adapters
        gateway = ctx.resources.gateway
        if gateway is None:
            message = "Storage gateway is required for CST extraction"
            raise ValueError(message)
        storage = self._storage_factory(gateway)
        discovery = self._discovery_factory(ctx.repo_root)

        # Get module paths and convert to ModuleRecord
        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

        # Execute step
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
