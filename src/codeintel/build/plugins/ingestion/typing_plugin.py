"""Typing ingest plugin.

This module provides `TypingIngestPlugin` that populates
analytics.typedness and analytics.static_diagnostics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.errors import GatewayNotAvailableError
from codeintel.build.plugin import FactoryPlugin
from codeintel.build.plugins.ingestion.helpers import get_module_paths, paths_to_modules
from codeintel.build.protocols import TypeChecker
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import (
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from codeintel.ingestion.ports.storage import IngestStoragePort

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.plugin import DiscoveryFactory, StorageFactory
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort

log = logging.getLogger(__name__)

TypeCheckerFactory = Callable[[TypeChecker | None], TypeChecker | None]
TypingStepFactory = Callable[
    [IngestStoragePort, "ModuleDiscoveryPort", BuildToolAdapter], TypingIngestStep
]


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


def _default_typing_step_factory(
    storage: IngestStoragePort,
    discovery: ModuleDiscoveryPort,
    tools: BuildToolAdapter,
) -> TypingIngestStep:
    """Create a TypingIngestStep with injected ports.

    Parameters
    ----------
    storage
        Storage port for persisting computed rows.
    discovery
        Discovery port for reading module sources.
    tools
        Tool adapter for running type checker and linter commands.

    Returns
    -------
    TypingIngestStep
        Configured typing ingestion step.
    """
    return TypingIngestStep(storage, discovery, tools)


def _default_type_checker_factory(checker: TypeChecker | None) -> TypeChecker | None:
    """Passthrough factory for default type checker injection.

    Returns
    -------
    TypeChecker | None
        Provided type checker unchanged.
    """
    return checker


class TypingIngestPlugin(FactoryPlugin[TypingIngestStep]):
    """Populate analytics.typedness and analytics.static_diagnostics.

    This plugin runs type checkers (pyright, pyrefly) and linters (ruff)
    to compute typedness scores and capture static diagnostics.

    Outputs
    -------
    - analytics.typedness: Type coverage metrics
    - analytics.static_diagnostics: Static analysis diagnostics
    """

    _core_metadata: ClassVar[CorePluginMetadata] = TYPING_INGEST_METADATA

    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[TypingStepFactory] = _default_typing_step_factory

    _type_checker_factory: TypeCheckerFactory

    def __init__(
        self,
        *,
        storage_adapter_factory: StorageFactory | None = None,
        discovery_adapter_factory: DiscoveryFactory | None = None,
        type_checker_factory: TypeCheckerFactory | None = None,
        step_factory: TypingStepFactory | None = None,
        options_resolver: PluginOptionsResolver | None = None,
    ) -> None:
        super().__init__(
            storage_adapter_factory=storage_adapter_factory,
            discovery_adapter_factory=discovery_adapter_factory,
            step_factory=step_factory,
            options_resolver=options_resolver,
        )
        self._type_checker_factory = type_checker_factory or _default_type_checker_factory

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
        GatewayNotAvailableError
            If no storage gateway is available.
        """
        type_checker = self._type_checker_factory(ctx.resources.type_checker)
        if type_checker is None:
            log.info("Type checker not available, skipping typing analysis")
            return TargetResult.succeeded(row_counts={})

        paths = get_module_paths(ctx)
        modules = paths_to_modules(paths, ctx.repo_root)

        gateway = ctx.resources.gateway
        if gateway is None:
            context = "typing ingest"
            raise GatewayNotAvailableError(context)
        storage = self._storage_factory(gateway)
        discovery = self._discovery_factory(ctx.repo_root)
        tools = BuildToolAdapter(type_checker=type_checker)

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
