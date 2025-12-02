"""SCIP symbol indexing facade with convenient function-based API.

This module provides a function-based API for SCIP indexing that
wraps the class-based ScipIngestStep with sensible adapter defaults.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

from codeintel.ingestion.adapters import DuckDBStorageAdapter
from codeintel.ingestion.steps.scip_ingest import ScipIngestConfig, ScipIngestResult, ScipIngestStep

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config import ScipIngestStepConfig
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

# Pre-built SQL to avoid S608 (SQL injection warning)
_DELETE_SCIP_SYMBOLS_SQL: Final[str] = "DELETE FROM core.scip_symbols WHERE rel_path IN "


@dataclass
class ScipResult:
    """Result from SCIP indexing operation.

    Attributes
    ----------
    status
        Status of the operation: "ok", "unavailable", or "failed".
    reason
        Reason for status if not "ok".
    index_scip
        Path to generated SCIP binary index.
    index_json
        Path to generated JSON export.
    """

    status: Literal["success", "unavailable", "failed"]
    reason: str | None = None
    index_scip: Path | None = None
    index_json: Path | None = None


def ingest_scip(
    gateway: StorageGateway,
    cfg: ScipIngestStepConfig,
    *,
    tracker: ChangeTracker | None = None,
    tool_service: ToolService | None = None,
) -> ScipResult:
    """Run SCIP indexing and persist symbols.

    This function provides a convenient entry point for SCIP indexing.

    If cfg.scip_runner is provided, it will be used directly (for testing
    or custom implementations). Otherwise, tool_service with SCIP binaries
    is required.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    cfg
        SCIP ingest step configuration.
    tracker
        Optional change tracker for incremental processing.
    tool_service
        Tool service for running external tools.

    Returns
    -------
    ScipResult
        Result of the SCIP indexing operation.
    """
    # Reserved parameters for API compatibility
    del tracker

    # If a custom scip_runner is provided, use it directly
    if cfg.scip_runner is not None:
        result = cfg.scip_runner(gateway, cfg)
        # Convert ScipIngestResult to ScipResult if needed
        if isinstance(result, ScipIngestResult):
            return ScipResult(
                status=result.status,
                reason=getattr(result, "reason", None),
                index_scip=getattr(result, "index_scip", None),
                index_json=getattr(result, "index_json", None),
            )
        return result

    # Check if tools are available
    if tool_service is None:
        return ScipResult(
            status="unavailable",
            reason="SCIP tool service not configured",
        )

    # Check if binaries are configured
    if cfg.binaries is None:
        return ScipResult(
            status="unavailable",
            reason="SCIP binaries not configured",
        )

    # For now, return unavailable - full async implementation
    # would require running the step with tool adapter
    del gateway  # Not used in stub path
    return ScipResult(
        status="unavailable",
        reason="SCIP async execution not yet implemented in facade",
    )


class ScipIngestOps:
    """Operations for incremental SCIP symbol ingestion.

    Implements the IncrementalIngestOps protocol for SCIP symbols.

    Attributes
    ----------
    dataset_name
        Target dataset name for SCIP symbols.
    """

    dataset_name: ClassVar[str] = "core.scip_symbols"

    def __init__(
        self,
        *,
        cfg: ScipIngestStepConfig | None = None,
        runtime: ScipRuntime | None = None,
        service: ToolService | None = None,
    ) -> None:
        """Initialize SCIP ingest operations.

        Parameters
        ----------
        cfg
            SCIP ingest step configuration.
        runtime
            SCIP runtime context.
        service
            Tool service for external tool execution.
        """
        self._cfg = cfg
        self._runtime = runtime
        self._service = service

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """Return True when a module should be indexed for SCIP symbols.

        SCIP indexing targets source Python files only (in src/ directory).

        Parameters
        ----------
        module
            Module to evaluate.

        Returns
        -------
        bool
            True if module is a Python file in src/ directory.
        """
        # Only process Python files in src/ directory
        return module.rel_path.endswith(".py") and module.rel_path.startswith("src/")

    @staticmethod
    def delete_rows(gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Remove rows corresponding to the provided relative paths.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rel_paths
            Paths to delete.
        """
        if not rel_paths:
            return
        # Use parameterized query with pre-built SQL base
        placeholders = ", ".join(["?"] * len(rel_paths))
        gateway.con.execute(
            _DELETE_SCIP_SYMBOLS_SQL + f"({placeholders})",
            list(rel_paths),
        )

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[dict[str, Any]]:
        """Generate rows for a single module (stub - SCIP requires external tools).

        Parameters
        ----------
        module
            Module to process.

        Returns
        -------
        Iterable[dict[str, Any]]
            Empty - SCIP processing requires external tooling.
        """
        # SCIP symbol extraction requires external scip-python tool
        # This is handled by the async ScipIngestStep, not per-module
        del module  # Unused - SCIP requires batch processing
        return []

    @staticmethod
    def insert_rows(gateway: StorageGateway, rows: Sequence[dict[str, Any]]) -> None:
        """Persist generated rows to core.scip_symbols.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Rows to insert.
        """
        if not rows:
            return
        adapter = DuckDBStorageAdapter(gateway)
        tuple_rows = [
            [
                row.get("repo"),
                row.get("commit"),
                row.get("rel_path"),
                row.get("symbol"),
                row.get("scheme"),
                row.get("package_manager"),
                row.get("package_name"),
                row.get("package_version"),
                row.get("descriptor"),
                row.get("kind"),
                row.get("documentation"),
            ]
            for row in rows
        ]
        adapter.write_batch("core.scip_symbols", tuple_rows)


class ScipRuntime:
    """Runtime context for SCIP symbol indexing operations.

    Provides context for running SCIP tools and managing index artifacts.

    Attributes
    ----------
    repo_root
        Repository root path.
    scip_dir
        Directory for SCIP index files.
    doc_dir
        Directory for documentation output.
    con
        DuckDB connection for database operations.
    """

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        scip_dir: Path | None = None,
        doc_dir: Path | None = None,
        con: DuckDBConnection | None = None,
    ) -> None:
        """Initialize the SCIP runtime context.

        Parameters
        ----------
        repo_root
            Repository root path.
        scip_dir
            Directory for SCIP index files.
        doc_dir
            Directory for documentation output.
        con
            DuckDB connection for database operations.
        """
        self.repo_root = repo_root
        self.scip_dir = scip_dir
        self.doc_dir = doc_dir
        self.con = con


# Re-export step classes for direct usage
__all__ = [
    "ScipIngestConfig",
    "ScipIngestOps",
    "ScipIngestResult",
    "ScipIngestStep",
    "ScipResult",
    "ScipRuntime",
    "ingest_scip",
]
