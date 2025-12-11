"""Typing ingest plugin.

This module provides `TypingIngestPlugin` that populates
analytics.typedness and analytics.static_diagnostics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from codeintel.build.plugin import TargetPlugin
from codeintel.build.protocols import TypeChecker
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import (
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
from codeintel.ingestion.ports.storage import IngestStoragePort
from codeintel.storage.ibis_types import filter_by

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
TypeCheckerFactory = Callable[[TypeChecker | None], TypeChecker | None]
StepFactory = Callable[[IngestStoragePort, ModuleDiscoveryPort, BuildToolAdapter], TypingIngestStep]


TYPING_INGEST_METADATA = CorePluginMetadata(
    name="ingest.typing",
    version="3.0.0",
    description="Populate analytics.typedness and analytics.static_diagnostics.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="typing",
    provides=("analytics.typedness", "analytics.static_diagnostics"),
    requires=("core.modules",),
    produces_tables=("analytics.typedness", "analytics.static_diagnostics"),
    consumes_tables=("core.modules",),
    supports_incremental=True,
    scope_aware=True,
    resource_hints={"requires_tools": ["pyright", "pyrefly", "ruff"]},
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
        stage=cast("PluginStage", core.stage or "typing"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


def _default_type_checker_factory(checker: TypeChecker | None) -> TypeChecker | None:
    """Passthrough factory for default type checker injection.

    Returns
    -------
    TypeChecker | None
        Provided type checker unchanged.
    """
    return checker


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
        df = filter_by(table, table.repo == ctx.repo, table.commit == ctx.commit).select("path")
        result = df.execute()
        return [str(path) for path in result["path"].tolist()]
    except (RuntimeError, OSError):
        return []


class TypingIngestPlugin(TargetPlugin):
    """Populate analytics.typedness and analytics.static_diagnostics.

    This plugin runs type checkers (pyright, pyrefly) and linters (ruff)
    to compute typedness scores and capture static diagnostics.

    Outputs
    -------
    - analytics.typedness: Type coverage metrics
    - analytics.static_diagnostics: Static analysis diagnostics
    """

    plugin_name: ClassVar[str] = "typing_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Populate analytics.typedness and analytics.static_diagnostics."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = TYPING_INGEST_METADATA

    # Class-level defaults for adapter and step factories
    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[StepFactory] = TypingIngestStep

    # Instance attributes (set in __init__)
    _storage_factory: StorageFactory
    _discovery_factory: DiscoveryFactory
    _type_checker_factory: TypeCheckerFactory
    _step_factory: StepFactory

    def __init__(
        self,
        *,
        storage_adapter_factory: StorageFactory | None = None,
        discovery_adapter_factory: DiscoveryFactory | None = None,
        type_checker_factory: TypeCheckerFactory | None = None,
        step_factory: StepFactory | None = None,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        self._storage_factory = storage_adapter_factory or type(self).default_storage_factory
        self._discovery_factory = discovery_adapter_factory or type(self).default_discovery_factory
        self._type_checker_factory = type_checker_factory or _default_type_checker_factory
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
        """Execute typing analysis.

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
        # Check if type checker is available (soft dependency)
        type_checker = self._type_checker_factory(ctx.resources.type_checker)
        if type_checker is None:
            log.info("Type checker not available, skipping typing analysis")
            return TargetResult.succeeded(row_counts={})

        # Get module paths and convert to ModuleRecord
        paths = _get_module_paths(ctx)
        modules = _paths_to_modules(paths, ctx.repo_root)

        # Create adapters using build protocols
        gateway = ctx.resources.gateway
        if gateway is None:
            message = "Storage gateway is required for typing ingest"
            raise ValueError(message)
        storage = self._storage_factory(gateway)
        discovery = self._discovery_factory(ctx.repo_root)
        tools = BuildToolAdapter(type_checker=type_checker)

        # Execute step
        step = self._step_factory(storage, discovery, tools)
        result = await step.execute_async(
            modules,
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=str(ctx.repo_root),
        )

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            log.warning("Typing ingest failed: %s", errors)
            return TargetResult.failed(f"Typing ingest failed: {errors}")

        return TargetResult.succeeded(row_counts=result.table_counts or {})


__all__ = [
    "TYPING_INGEST_METADATA",
    "TypingIngestPlugin",
]
