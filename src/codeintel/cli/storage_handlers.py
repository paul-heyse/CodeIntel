"""Typer-free storage validation helpers used by the Cyclopts CLI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Import consolidated setup_logging from handlers.base
from codeintel.cli.handlers.base import setup_logging as _setup_logging_impl
from codeintel.cli.results import CliResult
from codeintel.storage.gateway import StorageConfig, StorageConnectionError, open_gateway
from codeintel.storage.helpers.profiling import run_profile
from codeintel.storage.macros.generation import RenderedMacro, render_macro
from codeintel.storage.metadata import (
    _assert_macro_coverage,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)

if TYPE_CHECKING:
    from codeintel.cli.execution.context import ExecutionContext

LOG = logging.getLogger(__name__)


class MacroRequirement(Enum):
    """Policy for ingest macro validation."""

    REQUIRE = "require"
    ALLOW_MISSING = "allow_missing"


# Use consolidated setup_logging from handlers.base
setup_logging = _setup_logging_impl


def storage_validate_macros(
    db_path: Path,
    macro_requirement: MacroRequirement,
    verbose: int,
) -> None:
    """Validate macro registry hashes and normalized macro schemas.

    Raises
    ------
    RuntimeError
        If validation fails or required macros are missing.
    """
    setup_logging(verbose)

    try:
        gateway = open_gateway(StorageConfig.for_readonly(db_path))
    except StorageConnectionError as exc:
        LOG.warning("Falling back to existing database attachment: %s", exc)
        return
    connection = gateway.con
    missing_ingest: list[str] = []
    error: RuntimeError | None = None

    try:
        _assert_macro_coverage()
        validate_macro_registry(connection)
        validate_dataset_schema_registry(connection)
        validate_normalized_macro_schemas(connection)
        missing_ingest, present_ingest = ingest_macro_coverage(connection)
        if missing_ingest:
            LOG.warning("Missing ingest macros: %s", ", ".join(missing_ingest))
        LOG.debug("Present ingest macros: %s", ", ".join(present_ingest))
        if macro_requirement is MacroRequirement.REQUIRE and missing_ingest:
            message = ", ".join(missing_ingest)
            error = RuntimeError(f"Ingest macros missing: {message}")
    except RuntimeError as exc:
        error = exc

    if error is not None:
        gateway.close()
        raise RuntimeError(str(error)) from error

    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )

    gateway.close()


def generate_macros_for_tables(
    tables: list[str] | None,
    *,
    verbose: int,
) -> list[RenderedMacro]:
    """Render normalized macro DDL for the requested tables.

    Raises
    ------
    RuntimeError
        If no tables are available to render.

    Returns
    -------
    list[RenderedMacro]
        Rendered macro definitions.
    """
    setup_logging(verbose)

    def _iter_tables(selected: list[str] | None) -> list[str]:
        if selected:
            return list(selected)
        return []

    table_keys = _iter_tables(tables)
    if not table_keys:
        msg = "No tables available to render macros for."
        raise RuntimeError(msg)

    return [render_macro(table_key) for table_key in table_keys]


def profile_storage_paths(
    db_path: Path,
    output_dir: Path,
    *,
    include_views: bool = False,
    verbose: int = 0,
) -> None:
    """Run storage profiling."""
    setup_logging(verbose)
    run_profile(db_path=db_path, output_dir=output_dir, analyze=include_views)


# =============================================================================
# Result Types for Structured Handlers
# =============================================================================


@dataclass
class MacroValidationResult:
    """Result from macro validation.

    Parameters
    ----------
    status
        Validation status (valid, skipped, invalid).
    missing_ingest
        List of missing ingest macro names.
    present_ingest
        List of present ingest macro names.
    dataset_rows_only
        List of datasets with rows only (no normalized macro).
    reason
        Optional reason for status (e.g., skip reason).
    """

    status: str
    missing_ingest: list[str]
    present_ingest: list[str]
    dataset_rows_only: list[str]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        result: dict[str, Any] = {
            "status": self.status,
            "missing_ingest": self.missing_ingest,
            "present_ingest": self.present_ingest,
            "dataset_rows_only": self.dataset_rows_only,
        }
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass
class MacroGenerationResult:
    """Result from macro generation.

    Parameters
    ----------
    macros
        List of rendered macro definitions with macro_name and ddl.
    count
        Number of macros generated.
    """

    macros: list[dict[str, str]]
    count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "macros": self.macros,
            "count": self.count,
        }


@dataclass
class StorageProfileResult:
    """Result from storage profiling.

    Parameters
    ----------
    db_path
        Path to the profiled database.
    output_dir
        Directory where profile output was written.
    include_views
        Whether views were included in profiling.
    """

    db_path: str
    output_dir: str
    include_views: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "db_path": self.db_path,
            "output_dir": self.output_dir,
            "include_views": self.include_views,
        }


# =============================================================================
# Structured Handlers (ExecutionContext Pattern)
# =============================================================================


def storage_validate_macros_structured(
    ctx: ExecutionContext,
) -> CliResult[MacroValidationResult]:
    """Validate macro registry hashes and normalized macro schemas (structured).

    Parameters
    ----------
    ctx
        Execution context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - macro_requirement: MacroRequirement enum value

    Returns
    -------
    CliResult[MacroValidationResult]
        Validation result with status and any issues found.

    Raises
    ------
    RuntimeError
        If validation fails or required macros are missing.
    """
    setup_logging(ctx.verbosity)

    # Get db_path from params or runtime
    db_path = ctx.get_str_param("db_path")
    if db_path is None:
        runtime = ctx.require_runtime()
        db_path_resolved = runtime.db_path
    else:
        db_path_resolved = Path(db_path)

    macro_req_str = ctx.get_str_param("macro_requirement", "require")
    macro_requirement = MacroRequirement(macro_req_str)

    try:
        gateway = open_gateway(StorageConfig.for_readonly(db_path_resolved))
    except StorageConnectionError as exc:
        LOG.warning("Falling back to existing database attachment: %s", exc)
        return CliResult.ok(
            MacroValidationResult(
                status="skipped",
                missing_ingest=[],
                present_ingest=[],
                dataset_rows_only=[],
                reason=str(exc),
            )
        )

    connection = gateway.con
    missing_ingest: list[str] = []
    present_ingest: list[str] = []
    error: RuntimeError | None = None

    try:
        _assert_macro_coverage()
        validate_macro_registry(connection)
        validate_dataset_schema_registry(connection)
        validate_normalized_macro_schemas(connection)
        missing_ingest, present_ingest = ingest_macro_coverage(connection)
        if missing_ingest:
            LOG.warning("Missing ingest macros: %s", ", ".join(missing_ingest))
        LOG.debug("Present ingest macros: %s", ", ".join(present_ingest))
        if macro_requirement is MacroRequirement.REQUIRE and missing_ingest:
            message = ", ".join(missing_ingest)
            error = RuntimeError(f"Ingest macros missing: {message}")
    except RuntimeError as exc:
        error = exc

    if error is not None:
        gateway.close()
        raise RuntimeError(str(error)) from error

    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )

    gateway.close()

    return CliResult.ok(
        MacroValidationResult(
            status="valid",
            missing_ingest=missing_ingest,
            present_ingest=present_ingest,
            dataset_rows_only=dataset_rows_list,
        )
    )


def generate_macros_structured(
    ctx: ExecutionContext,
) -> CliResult[MacroGenerationResult]:
    """Render normalized macro DDL for the requested tables (structured).

    Parameters
    ----------
    ctx
        Execution context with params:
        - tables: List of table names to generate macros for

    Returns
    -------
    CliResult[MacroGenerationResult]
        Generated macro definitions.

    Raises
    ------
    RuntimeError
        If no tables are available to render.
    """
    setup_logging(ctx.verbosity)

    # Get tables from params - use raw params dict for list access
    tables_raw = ctx.params.get("tables")
    tables: list[str] = list(tables_raw) if tables_raw else []

    if not tables:
        msg = "No tables available to render macros for."
        raise RuntimeError(msg)

    rendered = [render_macro(table_key) for table_key in tables]
    macro_dicts = [
        {"macro_name": m.macro_name, "ddl": m.ddl} for m in rendered
    ]

    return CliResult.ok(
        MacroGenerationResult(
            macros=macro_dicts,
            count=len(rendered),
        )
    )


def profile_storage_structured(
    ctx: ExecutionContext,
) -> CliResult[StorageProfileResult]:
    """Run storage profiling (structured).

    Parameters
    ----------
    ctx
        Execution context with params:
        - db_path: Path to database
        - output_dir: Output directory for profile results
        - include_views: Whether to include views in profiling

    Returns
    -------
    CliResult[StorageProfileResult]
        Profiling result with paths and options used.
    """
    setup_logging(ctx.verbosity)

    # Get db_path from params or runtime
    db_path_str = ctx.get_str_param("db_path")
    if db_path_str is None:
        runtime = ctx.require_runtime()
        db_path = runtime.db_path
    else:
        db_path = Path(db_path_str)

    output_dir_str = ctx.require_str_param("output_dir")
    output_dir = Path(output_dir_str)
    include_views = ctx.get_bool_param("include_views", default=False)

    run_profile(db_path=db_path, output_dir=output_dir, analyze=include_views)

    return CliResult.ok(
        StorageProfileResult(
            db_path=str(db_path),
            output_dir=str(output_dir),
            include_views=include_views,
        )
    )


__all__ = [
    "MacroGenerationResult",
    "MacroRequirement",
    "MacroValidationResult",
    "StorageProfileResult",
    "generate_macros_for_tables",
    "generate_macros_structured",
    "profile_storage_paths",
    "profile_storage_structured",
    "storage_validate_macros",
    "storage_validate_macros_structured",
]
