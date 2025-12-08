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
from codeintel.build.result import TargetResult
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
)
from codeintel.ingestion.compute import AstExtractStep
from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
from codeintel.ingestion.ports.storage import IngestStoragePort

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway
else:
    StorageGateway = object

log = logging.getLogger(__name__)

StorageFactory = Callable[[StorageGateway], IngestStoragePort]
DiscoveryFactory = Callable[[Path], ModuleDiscoveryPort]
StepFactory = Callable[[IngestStoragePort, ModuleDiscoveryPort], AstExtractStep]


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

    # Class-level defaults for adapter and step factories
    default_storage_factory: ClassVar[StorageFactory] = DuckDBStorageAdapter
    default_discovery_factory: ClassVar[DiscoveryFactory] = FilesystemDiscoveryAdapter
    default_step_factory: ClassVar[StepFactory] = AstExtractStep

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
    ) -> None:
        self._storage_factory = storage_adapter_factory or type(self).default_storage_factory
        self._discovery_factory = discovery_adapter_factory or type(self).default_discovery_factory
        self._step_factory = step_factory or type(self).default_step_factory

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
        # Create adapters
        gateway = ctx.resources.gateway
        if gateway is None:
            message = "Storage gateway is required for AST extraction"
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
    # First check if modules are provided in resources
    if ctx.resources.modules:
        return list(ctx.resources.modules)

    # Otherwise query from database
    try:
        rows = ctx.gateway.con.execute(
            "SELECT path FROM core.modules WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchall()
        return [str(row[0]) for row in rows]
    except (RuntimeError, OSError):
        return []


__all__ = ["AstExtractPlugin"]
