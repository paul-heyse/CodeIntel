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
from time import perf_counter
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    CacheLogIngestSummary,
    ProfileStorageResult,
    StorageDatabaseExportResult,
    StorageDatabaseImportResult,
    ValidateMacrosResult,
)
from codeintel.cli.errors import ValidationError, validation_error
from codeintel.cli.errors.results import (
    fail_macro_validation,
    fail_missing_output_path,
    fail_missing_required,
    fail_storage,
    fail_storage_connection,
)
from codeintel.core.errors.storage import StorageConnectionError
from codeintel.core.errors.taxonomy import INVALID_FORMAT
from codeintel.observability.cache_log_ingest import (
    CacheLogIngestConfigError,
    ingest_cache_log_jsonl,
)
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.contracts.provider import iter_contracts
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.query_results import iter_tuples_from_arrow_reader
from codeintel.storage.validation import ContractValidationMode
from codeintel.storage.warehouse import Warehouse

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.cli.context import CommandContext
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.protocol import DuckDBConnection

LOG = logging.getLogger(__name__)


def _resolve_validation_mode(raw: str | None) -> ContractValidationMode:
    if raw is None:
        return ContractValidationMode.STRICT
    normalized = raw.lower()
    try:
        return ContractValidationMode(normalized)
    except ValueError as exc:
        msg = 'Invalid value for "--validation-mode"'
        raise ValidationError(msg) from exc


@contextmanager
def _readonly_gateway(
    db_path: Path,
    *,
    validation_mode: ContractValidationMode = ContractValidationMode.LENIENT,
) -> Iterator[StorageGateway]:
    """Open a read-only gateway with automatic cleanup.

    Parameters
    ----------
    db_path
        Path to the database.
    validation_mode
        Contract validation behavior when opening the gateway.

    Yields
    ------
    StorageGateway
        Open gateway that closes on context exit.
    """
    gw = open_gateway(StorageConfig.for_readonly(db_path, validation_mode=validation_mode))
    try:
        yield gw
    finally:
        gw.close()


@contextmanager
def _gateway_for_import(db_path: Path) -> Iterator[MinimalStorageGateway]:
    cfg = StorageConfig(
        db_path=db_path,
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    session = DuckDBSession(cfg)
    con = session.open()
    gw = MinimalStorageGateway(con)
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
    try:
        validation_mode = _resolve_validation_mode(ctx.params.get_str("validation_mode"))
    except ValidationError as exc:
        return CliResult.fail(
            validation_error(
                INVALID_FORMAT,
                "validation_mode",
                str(exc),
            )
        )

    if db_path_str is not None:
        db_path = Path(db_path_str)
        try:
            with _readonly_gateway(
                db_path,
                validation_mode=validation_mode,
            ) as gateway:
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


def _load_table_schema_registry_keys(connection: DuckDBConnection) -> set[str]:
    table_ref = meta_table_ref("metadata.table_schema_registry")
    reader = connection.execute(
        f"SELECT table_key FROM {table_ref}"
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    return {str(row[0]) for row in iter_tuples_from_arrow_reader(reader)}


def _load_missing_schema_versions(connection: DuckDBConnection) -> list[str]:
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    versions_ref = meta_table_ref("metadata.schema_versions")
    reader = connection.execute(
        f"""
        SELECT r.table_key
        FROM {registry_ref} AS r
        LEFT JOIN {versions_ref} AS v
          ON r.schema_digest = v.schema_digest
        WHERE v.schema_digest IS NULL
        """
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    return [str(row[0]) for row in iter_tuples_from_arrow_reader(reader)]


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

    try:
        expected_keys = {contract.table_key for contract in iter_contracts()}
    except RuntimeError as exc:
        return fail_macro_validation(str(exc))

    registry_keys = _load_table_schema_registry_keys(connection)
    missing_registry = sorted(expected_keys - registry_keys)
    missing_versions = _load_missing_schema_versions(connection)

    if missing_registry or missing_versions:
        parts: list[str] = []
        if missing_registry:
            parts.append(f"Missing table schema registry entries: {', '.join(missing_registry)}")
        if missing_versions:
            parts.append(
                "Missing schema versions for table keys: " + ", ".join(sorted(missing_versions))
            )
        return fail_macro_validation("; ".join(parts))

    dataset_rows_list: list[str] = []

    return CliResult.ok(
        ValidateMacrosResult(
            status="valid",
            missing_ingest=missing_ingest,
            present_ingest=present_ingest,
            dataset_rows_only=dataset_rows_list,
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
    output_dir = _resolve_profile_output_dir(ctx)
    if output_dir is None:
        return fail_missing_output_path("output_dir")

    db_path = _resolve_profile_db_path(ctx)
    include_views = ctx.params.get_bool("include_views", default=False)
    profile_gateway = _select_profile_gateway(ctx, db_path)

    views = ("docs.v_subsystem_profile", "docs.v_subsystem_coverage")
    if profile_gateway is not None:
        Warehouse(profile_gateway).profile_views(
            views=views,
            output_dir=output_dir,
            analyze=include_views,
            db_path=db_path,
        )
    else:
        with _readonly_gateway(db_path) as gateway:
            Warehouse(gateway).profile_views(
                views=views,
                output_dir=output_dir,
                analyze=include_views,
                db_path=db_path,
            )

    return CliResult.ok(
        ProfileStorageResult(
            db_path=str(db_path),
            output_dir=str(output_dir),
            include_views=include_views,
        )
    )


def ingest_cache_logs_handler(ctx: CommandContext) -> CliResult[CacheLogIngestSummary]:
    """Ingest Hamilton cache JSONL logs into DuckDB.

    Parameters
    ----------
    ctx
        Command context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - cache_dir: Cache directory to scan for JSONL files
        - jsonl_paths: Explicit JSONL files to ingest

    Returns
    -------
    CliResult[CacheLogIngestSummary]
        Ingestion summary payload.
    """
    db_path = _resolve_storage_db_path(ctx)
    cache_dir = ctx.params.get_path("cache_dir")
    jsonl_paths = _resolve_jsonl_paths(ctx)
    if cache_dir is None and not jsonl_paths:
        return fail_missing_required(
            "cache_dir",
            detail="Provide --cache-dir or at least one --jsonl-path.",
        )
    try:
        result = ingest_cache_log_jsonl(
            duckdb_path=db_path,
            cache_dir=cache_dir,
            jsonl_paths=jsonl_paths,
        )
    except CacheLogIngestConfigError as exc:
        return fail_missing_required("cache_dir", detail=str(exc))
    except DuckDBError as exc:
        return fail_storage(str(exc))
    return CliResult.ok(
        CacheLogIngestSummary(
            db_path=str(db_path),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            inserted_events=result.inserted_events,
            run_ids=list(result.run_ids),
            jsonl_files=list(result.jsonl_files),
        )
    )


def export_database_handler(
    ctx: CommandContext,
) -> CliResult[StorageDatabaseExportResult]:
    """Export the DuckDB database to a directory.

    Returns
    -------
    CliResult[StorageDatabaseExportResult]
        Export result payload.
    """
    output_dir = ctx.params.get_path("output_dir")
    if output_dir is None:
        return fail_missing_output_path("output_dir")
    db_path = ctx.params.get_path("db_path")
    start = perf_counter()

    if db_path is not None:
        try:
            session = DuckDBSession(StorageConfig.for_readonly(db_path))
            con = session.open_reader()
            gw = MinimalStorageGateway(con)
            try:
                gw.export_database(directory=output_dir)
            finally:
                gw.close()
        except DuckDBError as exc:
            LOG.warning("Failed to connect to database at %s: %s", db_path, exc)
            return fail_storage_connection(db_path, str(exc))
    else:
        ctx.gateway.export_database(directory=output_dir)

    duration = perf_counter() - start
    return CliResult.ok(
        StorageDatabaseExportResult(
            db_path=str(db_path or ctx.gateway.config.db_path),
            output_dir=str(output_dir),
            duration_seconds=duration,
        )
    )


def import_database_handler(
    ctx: CommandContext,
) -> CliResult[StorageDatabaseImportResult]:
    """Import a DuckDB database from a directory.

    Returns
    -------
    CliResult[StorageDatabaseImportResult]
        Import result payload.
    """
    input_dir = ctx.params.get_path("input_dir")
    if input_dir is None:
        return fail_missing_output_path("input_dir")
    if not input_dir.is_dir():
        return fail_missing_output_path("input_dir")

    db_path = ctx.params.get_path("db_path")
    start = perf_counter()

    if db_path is not None:
        try:
            with _gateway_for_import(db_path) as gw:
                gw.import_database(directory=input_dir)
        except DuckDBError as exc:
            LOG.warning("Failed to connect to database at %s: %s", db_path, exc)
            return fail_storage_connection(db_path, str(exc))
    else:
        ctx.gateway.import_database(directory=input_dir)

    duration = perf_counter() - start
    return CliResult.ok(
        StorageDatabaseImportResult(
            db_path=str(db_path or ctx.gateway.config.db_path),
            input_dir=str(input_dir),
            duration_seconds=duration,
        )
    )


def _resolve_profile_output_dir(ctx: CommandContext) -> Path | None:
    output_dir_str = ctx.params.get_str("output_dir")
    if output_dir_str is None:
        return None
    return Path(output_dir_str)


def _resolve_storage_db_path(ctx: CommandContext) -> Path:
    db_path = ctx.params.get_path("db_path")
    if db_path is not None:
        return db_path
    if ctx.has_runtime:
        return ctx.runtime.paths.db_path
    gateway_db_path = getattr(getattr(ctx.gateway, "config", None), "db_path", None)
    if isinstance(gateway_db_path, (str, Path)):
        return Path(gateway_db_path)
    return Path(":memory:")


def _resolve_profile_db_path(ctx: CommandContext) -> Path:
    return _resolve_storage_db_path(ctx)


def _resolve_jsonl_paths(ctx: CommandContext) -> list[Path] | None:
    values = ctx.params.get_list("jsonl_paths")
    if not values:
        return None
    return [Path(value) for value in values]


def _select_profile_gateway(ctx: CommandContext, db_path: Path) -> StorageGateway | None:
    gateway_db_path = getattr(getattr(ctx.gateway, "config", None), "db_path", None)
    if not isinstance(gateway_db_path, (str, Path)):
        return None
    if str(db_path) == ":memory:" or str(gateway_db_path) == str(db_path):
        return ctx.gateway
    try:
        if Path(gateway_db_path).resolve() == db_path.resolve():
            return ctx.gateway
    except OSError:
        return None
    return None


__all__ = [
    "CacheLogIngestSummary",
    "ProfileStorageResult",
    "StorageDatabaseExportResult",
    "StorageDatabaseImportResult",
    "ValidateMacrosResult",
    "export_database_handler",
    "import_database_handler",
    "ingest_cache_logs_handler",
    "profile_storage_handler",
    "validate_macros_handler",
]
