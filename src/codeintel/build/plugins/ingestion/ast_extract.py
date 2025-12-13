"""AST extraction plugin.

This module provides `AstExtractPlugin` that parses Python AST
and persists rows + metrics into core.ast_* tables.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins._metadata import to_plugin_metadata
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute import AstExtractStep
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
from codeintel.ingestion.ports.storage import IngestStoragePort
from codeintel.storage.ibis_types import ibis_bool

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.storage.gateway import StorageGateway
else:
    StorageGateway = object

log = logging.getLogger(__name__)

StorageFactory = Callable[[StorageGateway], IngestStoragePort]
DiscoveryFactory = Callable[[Path], ModuleDiscoveryPort]
StepFactory = Callable[[IngestStoragePort, ModuleDiscoveryPort], AstExtractStep]


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


def _paths_to_modules(paths: list[str], repo_root: Path) -> list[ModuleRecord]:
    """Convert string paths to ModuleRecord objects.

    Parameters
    ----------
    paths
        List of relative file paths.
    repo_root
        Repository root directory.

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


class AstExtractPlugin(TargetPlugin):
    """Parse Python AST and persist rows + metrics.

    This plugin parses Python source files using the stdlib AST module,
    extracting node information and computing metrics.

    Outputs
    -------
    - core.ast_nodes: AST node information
    - core.ast_metrics: File-level AST metrics
    """

    plugin_name: ClassVar[str] = "ast_extract"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Parse Python AST and persist rows + metrics into core.ast_* tables."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = AST_EXTRACT_METADATA

    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[StepFactory] = AstExtractStep

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
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata definition."""
        return self._core_metadata

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
        ValueError
            If no storage gateway is available.
        """
        gateway = ctx.resources.gateway
        if gateway is None:
            message = "Storage gateway is required for AST extraction"
            raise ValueError(message)
        storage = self._storage_factory(gateway)
        discovery = self._discovery_factory(ctx.repo_root)

        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

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


def _get_module_paths(ctx: TargetExecutionContext) -> list[str]:
    """Get module paths from context resources or database.

    Parameters
    ----------
    ctx
        Execution context.

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
            table.filter(
                [
                    ibis_bool(table.repo == ctx.repo),
                    ibis_bool(table.commit == ctx.commit),
                ]
            )
            .select("path")
            .execute()
        )
        return [str(path) for path in df["path"].tolist()]
    except (RuntimeError, OSError):
        return []


__all__ = [
    "AST_EXTRACT_METADATA",
    "AstExtractPlugin",
]
