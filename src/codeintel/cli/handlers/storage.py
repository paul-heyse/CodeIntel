"""Storage handlers.

Handlers for storage validation, macro generation, and profiling operations.

These handlers support both runtime-resolved databases (via ctx.gateway)
and explicit database paths (via the db_path parameter). When an explicit db_path
is provided, the handler opens a dedicated gateway for that path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    GenerateMacrosResult,
    ProfileStorageResult,
    ValidateMacrosResult,
)
from codeintel.cli.errors.results import (
    fail_macro_validation,
    fail_missing_output_path,
    fail_no_tables,
)
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

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway import StorageGateway

LOG = logging.getLogger(__name__)


@contextmanager
def _readonly_gateway(db_path: Path) -> Iterator[StorageGateway]:
    """Open a read-only gateway with automatic cleanup.

    Parameters
    ----------
    db_path
        Path to the database.

    Yields
    ------
    StorageGateway
        Open gateway that closes on context exit.
    """
    gw = open_gateway(StorageConfig.for_readonly(db_path))
    try:
        yield gw
    finally:
        gw.close()


class MacroRequirement(Enum):
    """Policy for ingest macro validation."""

    REQUIRE = "require"
    ALLOW_MISSING = "allow_missing"


def validate_macros_handler(
    ctx: CommandContext,
) -> CliResult[ValidateMacrosResult]:
    """Validate macro registry hashes and normalized macro schemas.

    Parameters
    ----------
    ctx
        Command context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - macro_requirement: MacroRequirement enum value or string

    Returns
    -------
    CliResult[ValidateMacrosResult]
        Validation result with status and any issues found.

    Notes
    -----
    Uses explicit gateway when db_path is provided, otherwise uses ctx.gateway.
    """
    db_path_str = ctx.params.get_str("db_path")
    macro_requirement = (
        ctx.params.get_enum("macro_requirement", MacroRequirement, MacroRequirement.REQUIRE)
        or MacroRequirement.REQUIRE
    )

    # Determine which gateway to use
    if db_path_str is not None:
        # Explicit db_path provided - use dedicated gateway
        db_path = Path(db_path_str)
        try:
            with _readonly_gateway(db_path) as gateway:
                return _validate_macros(gateway, macro_requirement)
        except StorageConnectionError as exc:
            LOG.warning("Failed to connect to database at %s: %s", db_path, exc)
            return CliResult.ok(
                ValidateMacrosResult(
                    status="skipped",
                    missing_ingest=[],
                    present_ingest=[],
                    dataset_rows_only=[],
                    reason=str(exc),
                )
            )
    else:
        # No explicit path - use CommandContext's gateway (runtime-resolved)
        return _validate_macros(ctx.gateway, macro_requirement)


def _validate_macros(
    gateway: StorageGateway,
    macro_requirement: MacroRequirement,
) -> CliResult[ValidateMacrosResult]:
    """Perform macro validation against a gateway.

    Parameters
    ----------
    gateway
        Open storage gateway.
    macro_requirement
        Policy for missing ingest macros.

    Returns
    -------
    CliResult[ValidateMacrosResult]
        Validation result.
    """
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

    if error_msg is not None:
        return fail_macro_validation(error_msg)

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
    ctx: CommandContext,
) -> CliResult[GenerateMacrosResult]:
    """Render normalized macro DDL for the requested tables.

    Parameters
    ----------
    ctx
        Command context with params:
        - tables: List of table names to generate macros for

    Returns
    -------
    CliResult[GenerateMacrosResult]
        Generated macro definitions.
    """
    # Get tables from params
    tables = ctx.params.get_list("tables")

    if not tables:
        return fail_no_tables("No tables available to render macros for.")

    rendered = [render_macro(table_key) for table_key in tables]
    macro_dicts = [{"macro_name": m.macro_name, "ddl": m.ddl} for m in rendered]

    return CliResult.ok(
        GenerateMacrosResult(
            macros=macro_dicts,
            count=len(rendered),
        )
    )


def profile_storage_handler(
    ctx: CommandContext,
) -> CliResult[ProfileStorageResult]:
    """Run storage profiling.

    Parameters
    ----------
    ctx
        Command context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - output_dir: Output directory for profile results
        - include_views: Whether to include views in profiling

    Returns
    -------
    CliResult[ProfileStorageResult]
        Profiling result with paths and options used.
    """
    output_dir_str = ctx.params.get_str("output_dir")
    if output_dir_str is None:
        return fail_missing_output_path("output_dir")
    output_dir = Path(output_dir_str)

    db_path_str = ctx.params.get_str("db_path")
    if db_path_str is not None:
        db_path = Path(db_path_str)
    elif ctx.has_runtime:
        db_path = ctx.runtime.paths.db_path
    else:
        gateway_db_path = getattr(getattr(ctx.gateway, "config", None), "db_path", None)
        if isinstance(gateway_db_path, (str, Path)):
            db_path = Path(gateway_db_path)
        else:
            db_path = Path(":memory:")

    include_views = ctx.params.get_bool("include_views", default=False)

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
