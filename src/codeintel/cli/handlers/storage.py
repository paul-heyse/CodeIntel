"""Storage handlers.

Handlers for storage validation, macro generation, and profiling operations.

These handlers support both runtime-resolved databases (via ctx.gateway)
and explicit database paths (via the db_path parameter). When an explicit db_path
is provided, the handler opens a dedicated gateway for that path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.core.columnar import stream_from_items
from codeintel.cli.core.result_types import (
    CacheLogIngestSummary,
    ProfileStorageResult,
    StorageDatabaseExportResult,
    StorageDatabaseImportResult,
    TabularResult,
    ValidateMacrosResult,
)
from codeintel.cli.errors import ValidationError, validation_error
from codeintel.cli.errors.results import (
    fail_macro_validation,
    fail_missing_output_path,
    fail_missing_required,
    fail_not_found,
    fail_storage,
    fail_storage_connection,
)
from codeintel.cli.handlers.metadata_bundle import BundleIngestRequest, ingest_metadata_bundle
from codeintel.cli.rendering.types import OutputFormat
from codeintel.cli.services.storage import StorageService
from codeintel.core.columnar.conversion import tabular_to_arrow_reader
from codeintel.core.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.core.errors.storage import StorageConnectionError
from codeintel.core.errors.taxonomy import INVALID_FORMAT
from codeintel.core.storage import StorageContext
from codeintel.observability.cache_log_ingest import (
    CacheLogIngestConfigError,
    ingest_cache_log_jsonl,
)
from codeintel.storage.contracts.provider import iter_contracts
from codeintel.storage.gateway.protocol import DuckDBError
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from codeintel.storage.query_results import iter_tuples_from_arrow_reader
from codeintel.storage.validation import ContractValidationMode
from codeintel.storage.warehouse import Warehouse

if TYPE_CHECKING:
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
        service = StorageService.from_path(db_path, validation_mode=validation_mode)
        try:
            with service.gateway_scope(
                read_only=True,
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
        try:
            with ctx.storage.gateway_scope(
                read_only=True,
                validation_mode=validation_mode,
            ) as gateway:
                return _validate_macros(gateway)
        except StorageConnectionError as exc:
            LOG.warning("Failed to connect to storage gateway: %s", exc)
            return CliResult.ok(
                ValidateMacrosResult(
                    status="skipped",
                    missing_ingest=[],
                    present_ingest=[],
                    dataset_rows_only=[],
                    reason=str(exc),
                )
            )


def _load_table_schema_registry_keys(connection: DuckDBConnection) -> set[str]:
    table_ref = meta_table_ref("metadata.table_schema_registry")
    reader = tabular_to_arrow_reader(
        connection.execute(f"SELECT table_key FROM {table_ref}"),
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
    )
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
    )
    reader = tabular_to_arrow_reader(
        reader,
        batch_size=DEFAULT_ARROW_BATCH_SIZE,
    )
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

    views = ("docs.v_subsystem_profile",)
    if profile_gateway is not None:
        Warehouse(context=StorageContext(gateway=profile_gateway)).profile_views(
            views=views,
            output_dir=output_dir,
            analyze=include_views,
            db_path=db_path,
        )
    else:
        service = StorageService.from_path(db_path)
        with service.gateway_scope(read_only=True) as gateway:
            Warehouse(context=StorageContext(gateway=gateway)).profile_views(
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


def ingest_cache_logs_handler(
    ctx: CommandContext,
) -> CliResult[CacheLogIngestSummary | TabularResult]:
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
    CliResult[CacheLogIngestSummary | TabularResult]
        Ingestion summary payload or streamed JSONL file rows.
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
    summary = CacheLogIngestSummary(
        db_path=str(db_path),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        inserted_events=result.inserted_events,
        run_ids=list(result.run_ids),
        jsonl_files=list(result.jsonl_files),
    )
    if ctx.output_format == OutputFormat.JSONL:
        rows = [{"jsonl_file": path} for path in summary.jsonl_files]
        stream = stream_from_items(rows)
        return CliResult.ok(
            TabularResult(
                stream=stream,
                metadata={
                    "db_path": summary.db_path,
                    "cache_dir": summary.cache_dir,
                    "inserted_events": summary.inserted_events,
                    "run_ids": summary.run_ids,
                },
            )
        )
    return CliResult.ok(summary)


def ingest_metadata_bundle_handler(ctx: CommandContext) -> CliResult[dict[str, object]]:
    """Ingest build metadata bundles into the DuckDB meta catalog.

    Parameters
    ----------
    ctx
        Command context with params:
        - db_path: Path to database (optional, uses runtime if not provided)
        - bundle_root: Build metadata bundle root (defaults to build/metadata)

    Returns
    -------
    CliResult[dict[str, object]]
        Summary payload for the ingest operation.
    """
    bundle_root = ctx.params.get_path("bundle_root")
    if bundle_root is None:
        if not ctx.has_runtime:
            return fail_missing_required("bundle_root")
        bundle_root = ctx.runtime.paths.build_dir / "metadata"
    if not bundle_root.exists():
        return fail_not_found(
            "bundle_root",
            str(bundle_root),
            suggestion="Run `codeintel build run --all` to generate build metadata",
        )

    if not ctx.has_runtime:
        return fail_missing_required("db_path")
    snapshot = ctx.runtime.snapshot
    if not snapshot.repo or not snapshot.commit:
        return fail_storage("repo and commit must be set for storage.ingest-metadata")

    db_path = _resolve_storage_db_path(ctx)
    try:
        outcome = ingest_metadata_bundle(
            BundleIngestRequest(
                bundle_root=bundle_root,
                db_path=db_path,
                repo=snapshot.repo,
                commit=snapshot.commit,
                catalog_inputs={"source": "storage.ingest-metadata"},
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return fail_storage(str(exc))

    payload = outcome.report.to_payload()
    payload.update(
        {
            "run_id": outcome.run_id,
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "bundle_root": str(bundle_root),
            "bundle_manifest_repo": outcome.bundle_manifest.repo,
            "bundle_manifest_commit": outcome.bundle_manifest.commit,
            "bundle_manifest_run_id": outcome.bundle_manifest.run_id,
            "renderer_cache_backfill_rows": outcome.renderer_cache_rows,
        }
    )
    if outcome.override_refresh is None:
        payload.update(
            {
                "override_refresh_status": "skipped",
                "override_refresh_reason": "schema_manifest_missing",
                "override_refresh_version_id": None,
                "override_refresh_tables": 0,
                "override_refresh_schema_versions_rows": 0,
                "override_refresh_versions_rows": 0,
                "override_refresh_registry_rows": 0,
            }
        )
    else:
        payload.update(
            {
                "override_refresh_status": outcome.override_refresh.status,
                "override_refresh_reason": outcome.override_refresh.reason,
                "override_refresh_version_id": outcome.override_refresh.version_id,
                "override_refresh_tables": outcome.override_refresh.tables,
                "override_refresh_schema_versions_rows": outcome.override_refresh.schema_versions_rows,
                "override_refresh_versions_rows": outcome.override_refresh.override_versions_rows,
                "override_refresh_registry_rows": outcome.override_refresh.override_registry_rows,
            }
        )
    return CliResult.ok(payload)


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
            service = StorageService.from_path(
                db_path,
                validation_mode=ContractValidationMode.OFF,
            )
            with service.gateway_scope(
                read_only=True,
                validation_mode=ContractValidationMode.OFF,
            ) as gateway:
                gateway.export_database(directory=output_dir)
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
            service = StorageService.from_path(
                db_path,
                validation_mode=ContractValidationMode.OFF,
            )
            with service.gateway_scope(
                read_only=False,
                validation_mode=ContractValidationMode.OFF,
            ) as gateway:
                gateway.import_database(directory=input_dir)
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
    if ctx.has_storage:
        return ctx.storage.db_path
    if ctx.has_runtime:
        return ctx.runtime.paths.db_path
    return Path(":memory:")


def _resolve_profile_db_path(ctx: CommandContext) -> Path:
    return _resolve_storage_db_path(ctx)


def _resolve_jsonl_paths(ctx: CommandContext) -> list[Path] | None:
    values = ctx.params.get_list("jsonl_paths")
    if not values:
        return None
    return [Path(value) for value in values]


def _select_profile_gateway(ctx: CommandContext, db_path: Path) -> StorageGateway | None:
    if not ctx.has_storage:
        return None
    if str(db_path) == ":memory:":
        return ctx.gateway
    try:
        if ctx.storage.db_path.resolve() == db_path.resolve():
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
    "ingest_metadata_bundle_handler",
    "profile_storage_handler",
    "validate_macros_handler",
]
