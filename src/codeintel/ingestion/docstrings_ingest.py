"""Docstring extraction facade.

This module re-exports the docstring extraction functionality for backward compatibility
with imports that expect `codeintel.ingestion.docstrings_ingest`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final

from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.common import iter_modules
from codeintel.ingestion.steps.docstrings_extract import (
    DocstringContext,
    DocstringsExtractStep,
    DocstringVisitor,
    ParsedDocstring,
)
from codeintel.storage.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.config.steps_ingestion import DocstringStepConfig
    from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway

# Pre-built SQL to avoid S608 (SQL injection warning)
_DELETE_DOCSTRINGS_SQL: Final[str] = "DELETE FROM core.docstrings WHERE rel_path IN "


class DocstringIngestOps:
    """Operations for incremental docstring ingestion.

    Implements the IncrementalIngestOps protocol for docstrings.

    Attributes
    ----------
    dataset_name
        Target dataset name for docstrings.
    """

    dataset_name: ClassVar[str] = "core.docstrings"

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """Return True when a module should be processed for docstrings.

        Parameters
        ----------
        module
            Module to evaluate.

        Returns
        -------
        bool
            True if module is a Python file.
        """
        return module.rel_path.endswith(".py")

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
            _DELETE_DOCSTRINGS_SQL + f"({placeholders})",
            list(rel_paths),
        )

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[dict[str, Any]]:
        """Generate docstring rows for a single module.

        Parameters
        ----------
        module
            Module to process.

        Returns
        -------
        Iterable[dict[str, Any]]
            Extracted docstring rows.
        """
        # Docstring extraction is handled by DocstringsExtractStep
        # This stub is for protocol compliance
        del module  # Docstring extraction uses the step directly
        return []

    @staticmethod
    def insert_rows(gateway: StorageGateway, rows: Sequence[dict[str, Any]]) -> None:
        """Persist generated rows to core.docstrings.

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
                row.get("module"),
                row.get("qualname"),
                row.get("kind"),
                row.get("start_line"),
                row.get("end_line"),
                row.get("raw"),
                row.get("summary"),
                row.get("style"),
                row.get("params_json"),
                row.get("returns_json"),
                row.get("raises_json"),
                row.get("created_at"),
            ]
            for row in rows
        ]
        adapter.write_batch("core.docstrings", tuple_rows)


def ingest_docstrings(
    gateway: StorageGateway,
    cfg: DocstringStepConfig,
    *,
    code_profile: ScanProfile | None = None,
) -> None:
    """Ingest docstrings from repository.

    Extracts structured docstrings from all modules registered in core.modules.

    Parameters
    ----------
    gateway
        Storage gateway for database operations.
    cfg
        Docstring step configuration with snapshot info.
    code_profile
        Optional scan profile for filtering modules (unused, modules come from inventory).
    """
    del code_profile  # Modules come from inventory, not filesystem scan

    # Load module inventory
    module_map = load_module_map(
        gateway,
        cfg.snapshot.repo,
        cfg.snapshot.commit,
        language="python",
    )

    if not module_map:
        return  # No modules to process

    # Build module records from inventory
    modules = list(iter_modules(module_map, cfg.snapshot.repo_root))

    if not modules:
        return  # No modules found

    # Create adapters
    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(cfg.snapshot.repo_root)

    # Execute docstring extraction
    step = DocstringsExtractStep(storage=storage, discovery=discovery)
    step.execute(
        modules=modules,
        repo=cfg.snapshot.repo,
        commit=cfg.snapshot.commit,
    )


__all__ = [
    "DocstringContext",
    "DocstringIngestOps",
    "DocstringVisitor",
    "DocstringsExtractStep",
    "ParsedDocstring",
    "ingest_docstrings",
]
