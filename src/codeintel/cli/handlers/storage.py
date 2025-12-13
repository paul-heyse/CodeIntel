"""Storage handlers.

Handlers for storage validation, macro generation, and profiling operations.

These handlers support both runtime-resolved databases (via ctx.gateway)
and explicit database paths (via the db_path parameter). When an explicit db_path
is provided, the handler opens a dedicated gateway for that path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
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
)
from codeintel.storage.gateway import StorageConfig, StorageConnectionError, open_gateway
from codeintel.storage.helpers.profiling import run_profile
from codeintel.storage.metadata import (
    validate_dataset_schema_registry,
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


def validate_macros_handler(
    ctx: CommandContext,
) -> CliResult[ValidateMacrosResult]:
    """Validate macro registry hashes and normalized macro schemas.

    Parameters
    ----------
    ctx
        Command context with params:
        - db_path: Path to database (optional, uses runtime if not provided)

    Returns
    -------
    CliResult[ValidateMacrosResult]
        Validation result with status and any issues found.

    Notes
    -----
    Uses explicit gateway when db_path is provided, otherwise uses ctx.gateway.
    """
    db_path_str = ctx.params.get_str("db_path")

    if db_path_str is not None:
        db_path = Path(db_path_str)
        try:
            with _readonly_gateway(db_path) as gateway:
                return _validate_macros(gateway)
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
        return _validate_macros(ctx.gateway)


def _validate_macros(
    gateway: StorageGateway,
) -> CliResult[ValidateMacrosResult]:
    """Perform macro validation against a gateway.

    Parameters
    ----------
    gateway
        Open storage gateway.

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
        validate_dataset_schema_registry(connection)
    except RuntimeError as exc:
        error_msg = str(exc)

    if error_msg is not None:
        return fail_macro_validation(error_msg)

    dataset_rows_list: list[str] = []

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
    tables = ctx.params.get_list("tables")

    _ = tables
    LOG.warning("storage.generate_macros is deprecated; macros are retired.")

    return CliResult.ok(
        GenerateMacrosResult(
            macros=[],
            count=0,
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

    profile_con = None
    gateway_db_path = getattr(getattr(ctx.gateway, "config", None), "db_path", None)
    if isinstance(gateway_db_path, (str, Path)):
        if str(db_path) == ":memory:" or str(gateway_db_path) == str(db_path):
            profile_con = ctx.gateway.con
        else:
            try:
                if Path(gateway_db_path).resolve() == db_path.resolve():
                    profile_con = ctx.gateway.con
            except OSError:
                profile_con = None

    run_profile(db_path=db_path, output_dir=output_dir, analyze=include_views, con=profile_con)

    return CliResult.ok(
        ProfileStorageResult(
            db_path=str(db_path),
            output_dir=str(output_dir),
            include_views=include_views,
        )
    )


__all__ = [
    "GenerateMacrosResult",
    "ProfileStorageResult",
    "ValidateMacrosResult",
    "generate_macros_handler",
    "profile_storage_handler",
    "validate_macros_handler",
]
