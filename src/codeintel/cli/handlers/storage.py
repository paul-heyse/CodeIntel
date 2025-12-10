"""Storage handlers.

Handlers for storage validation, macro generation, and profiling operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from codeintel.cli.core import CliResult
from codeintel.cli.errors import ProblemDetail
from codeintel.cli.handlers.context import HandlerContext
from codeintel.storage.gateway import StorageConfig, StorageConnectionError, open_gateway
from codeintel.storage.helpers.profiling import run_profile
from codeintel.storage.macros.generation import render_macro
from codeintel.storage.metadata import (
    _assert_macro_coverage,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)

LOG = logging.getLogger(__name__)


class MacroRequirement(Enum):
    """Policy for ingest macro validation."""

    REQUIRE = "require"
    ALLOW_MISSING = "allow_missing"


@dataclass(frozen=True)
class ValidateMacrosResult:
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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "status": self.status,
            "missing_ingest": self.missing_ingest,
            "present_ingest": self.present_ingest,
            "dataset_rows_only": self.dataset_rows_only,
        }
        if self.reason:
            result["reason"] = self.reason
        return result


@dataclass(frozen=True)
class GenerateMacrosResult:
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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "macros": self.macros,
            "count": self.count,
        }


@dataclass(frozen=True)
class ProfileStorageResult:
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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "db_path": self.db_path,
            "output_dir": self.output_dir,
            "include_views": self.include_views,
        }


def validate_macros_handler(
    ctx: HandlerContext,
) -> CliResult[ValidateMacrosResult]:
    """Validate macro registry hashes and normalized macro schemas.

    Parameters
    ----------
    ctx
        Handler context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - macro_requirement: MacroRequirement enum value or string

    Returns
    -------
    CliResult[ValidateMacrosResult]
        Validation result with status and any issues found.
    """
    # Get db_path from params or runtime
    db_path_str = ctx.param_str("db_path")
    db_path = ctx.runtime.paths.db_path if db_path_str is None else Path(db_path_str)

    macro_req_raw = ctx._params.get("macro_requirement", MacroRequirement.REQUIRE)
    if isinstance(macro_req_raw, MacroRequirement):
        macro_requirement = macro_req_raw
    else:
        macro_requirement = MacroRequirement(str(macro_req_raw))

    try:
        gateway = open_gateway(StorageConfig.for_readonly(db_path))
    except StorageConnectionError as exc:
        LOG.warning("Falling back to existing database attachment: %s", exc)
        return CliResult.ok(
            ValidateMacrosResult(
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
    error_msg: str | None = None

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
            error_msg = f"Ingest macros missing: {', '.join(missing_ingest)}"
    except RuntimeError as exc:
        error_msg = str(exc)

    gateway.close()

    if error_msg is not None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:storage:macro-validation-failed",
                title="Macro Validation Failed",
                detail=error_msg,
                status=422,
            )
        )

    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )

    return CliResult.ok(
        ValidateMacrosResult(
            status="valid",
            missing_ingest=missing_ingest,
            present_ingest=present_ingest,
            dataset_rows_only=dataset_rows_list,
        )
    )


def generate_macros_handler(
    ctx: HandlerContext,
) -> CliResult[GenerateMacrosResult]:
    """Render normalized macro DDL for the requested tables.

    Parameters
    ----------
    ctx
        Handler context with params:
        - tables: List of table names to generate macros for

    Returns
    -------
    CliResult[GenerateMacrosResult]
        Generated macro definitions.
    """
    # Get tables from params
    tables = ctx.param_list("tables")

    if not tables:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:storage:no-tables",
                title="No Tables Specified",
                detail="No tables available to render macros for.",
                status=400,
            )
        )

    rendered = [render_macro(table_key) for table_key in tables]
    macro_dicts = [{"macro_name": m.macro_name, "ddl": m.ddl} for m in rendered]

    return CliResult.ok(
        GenerateMacrosResult(
            macros=macro_dicts,
            count=len(rendered),
        )
    )


def profile_storage_handler(
    ctx: HandlerContext,
) -> CliResult[ProfileStorageResult]:
    """Run storage profiling.

    Parameters
    ----------
    ctx
        Handler context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - output_dir: Output directory for profile results
        - include_views: Whether to include views in profiling

    Returns
    -------
    CliResult[ProfileStorageResult]
        Profiling result with paths and options used.
    """
    # Get db_path from params or runtime
    db_path_str = ctx.param_str("db_path")
    db_path = ctx.runtime.paths.db_path if db_path_str is None else Path(db_path_str)

    output_dir_str = ctx.param_str("output_dir")
    if output_dir_str is None:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:storage:missing-output-dir",
                title="Missing Output Directory",
                detail="output_dir parameter is required.",
                status=400,
            )
        )
    output_dir = Path(output_dir_str)

    include_views = ctx.param_bool("include_views", default=False)

    run_profile(db_path=db_path, output_dir=output_dir, analyze=include_views)

    return CliResult.ok(
        ProfileStorageResult(
            db_path=str(db_path),
            output_dir=str(output_dir),
            include_views=include_views,
        )
    )


__all__ = [
    "GenerateMacrosResult",
    "MacroRequirement",
    "ProfileStorageResult",
    "ValidateMacrosResult",
    "generate_macros_handler",
    "profile_storage_handler",
    "validate_macros_handler",
]
