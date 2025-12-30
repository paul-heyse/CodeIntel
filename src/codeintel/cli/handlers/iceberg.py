"""Iceberg CLI handlers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.table.maintenance import MaintenanceTable
from pyiceberg.table.refs import SnapshotRefType

from codeintel.cli.core import CliResult
from codeintel.cli.core.result_types import (
    IcebergAddFilesResult,
    IcebergExpireSnapshotsResult,
    IcebergInspectResult,
    IcebergManageSnapshotsResult,
    IcebergRefreshCacheResult,
    IcebergRefsResult,
    IcebergTimeTravelResult,
)
from codeintel.cli.errors.results import (
    fail_invalid_value,
    fail_missing_required,
    fail_project_error,
)
from codeintel.core.config.view import SettingsView
from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.core.time import utc_now
from codeintel.storage.iceberg.cache import refresh_iceberg_metadata_cache
from codeintel.storage.iceberg.migration import IcebergAddFilesRequest, add_files_to_iceberg

if TYPE_CHECKING:
    from pyiceberg.table import Table

    from codeintel.cli.context import CommandContext
    from codeintel.core.config.settings import IcebergSettings


ICEBERG_HANDLER_ERRORS = (KeyError, OSError, RuntimeError, ValueError)
_REF_TYPES: tuple[str, str] = ("tag", "branch")


@dataclass(frozen=True, slots=True)
class _ManageSnapshotsRequest:
    table_key: str
    ref_name: str
    ref_type: Literal["tag", "branch"]
    remove_ref: bool
    snapshot_id: int | None


def iceberg_inspect_handler(ctx: CommandContext) -> CliResult[IcebergInspectResult | None]:
    """Inspect Iceberg table metadata.

    Returns
    -------
    CliResult[IcebergInspectResult | None]
        CLI result with inspection payload or error info.
    """
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_missing_required("table_key")
    snapshot_id = _optional_int(ctx, "snapshot_id")

    include_snapshots = ctx.params.get_bool("snapshots")
    include_manifests = ctx.params.get_bool("manifests")
    include_entries = ctx.params.get_bool("entries")
    include_refs = ctx.params.get_bool("refs")
    if not any([include_snapshots, include_manifests, include_entries, include_refs]):
        include_snapshots = True

    try:
        table = _load_table(table_key)
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.inspect", str(exc))

    inspect = table.inspect
    snapshots = _table_to_rows(inspect.snapshots()) if include_snapshots else None
    manifests = _table_to_rows(inspect.manifests()) if include_manifests else None
    entries = _table_to_rows(inspect.entries(snapshot_id=snapshot_id)) if include_entries else None
    refs = _table_to_rows(inspect.refs()) if include_refs else None

    return CliResult.ok(
        IcebergInspectResult(
            table_key=table_key,
            snapshots=snapshots,
            manifests=manifests,
            entries=entries,
            refs=refs,
        )
    )


def iceberg_refs_handler(ctx: CommandContext) -> CliResult[IcebergRefsResult | None]:
    """List Iceberg snapshot refs.

    Returns
    -------
    CliResult[IcebergRefsResult | None]
        CLI result with snapshot refs or error info.
    """
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_missing_required("table_key")
    try:
        table = _load_table(table_key)
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.refs", str(exc))
    refs = _table_to_rows(table.inspect.refs())
    return CliResult.ok(IcebergRefsResult(table_key=table_key, refs=refs))


def iceberg_manage_snapshots_handler(
    ctx: CommandContext,
) -> CliResult[IcebergManageSnapshotsResult | None]:
    """Create or remove Iceberg snapshot refs.

    Returns
    -------
    CliResult[IcebergManageSnapshotsResult | None]
        CLI result with ref mutation info or error details.
    """
    request = _parse_manage_snapshots_request(ctx)
    if isinstance(request, CliResult):
        return request

    try:
        table = _load_table(request.table_key)
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.manage_snapshots", str(exc))

    try:
        _apply_manage_snapshots(table, request=request)
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.manage_snapshots", str(exc))

    action = "removed" if request.remove_ref else "created"
    return CliResult.ok(
        IcebergManageSnapshotsResult(
            table_key=request.table_key,
            ref_name=request.ref_name,
            ref_type=request.ref_type,
            action=action,
            snapshot_id=None if request.remove_ref else request.snapshot_id,
        )
    )


def iceberg_expire_snapshots_handler(
    ctx: CommandContext,
) -> CliResult[IcebergExpireSnapshotsResult | None]:
    """Expire Iceberg snapshots using retention settings.

    Returns
    -------
    CliResult[IcebergExpireSnapshotsResult | None]
        CLI result with expiration details or error info.
    """
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_missing_required("table_key")
    retention_days = _optional_int(ctx, "retention_days")
    if retention_days is None or retention_days <= 0:
        return fail_invalid_value(
            "retention_days",
            retention_days,
            "retention_days must be a positive integer",
        )
    dry_run = ctx.params.get_bool("dry_run")
    confirm = ctx.params.get_bool("confirm")
    if not dry_run and not confirm:
        return fail_invalid_value(
            "confirm",
            confirm,
            "confirmation is required for snapshot expiration",
            suggestion="Pass --confirm or use --dry-run",
        )

    try:
        table = _load_table(table_key)
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.expire_snapshots", str(exc))

    cutoff = utc_now() - timedelta(days=retention_days)
    expired_ids = _expired_snapshot_ids(table, cutoff=cutoff)
    if not dry_run and expired_ids:
        MaintenanceTable(table).expire_snapshots().older_than(cutoff).commit()

    return CliResult.ok(
        IcebergExpireSnapshotsResult(
            table_key=table_key,
            cutoff=cutoff.isoformat(),
            expired_snapshot_ids=expired_ids,
            dry_run=dry_run,
        )
    )


def iceberg_time_travel_handler(
    ctx: CommandContext,
) -> CliResult[IcebergTimeTravelResult | None]:
    """Export an Iceberg snapshot to IPC or Parquet.

    Returns
    -------
    CliResult[IcebergTimeTravelResult | None]
        CLI result with export details or error info.
    """
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_missing_required("table_key")
    snapshot_id = _optional_int(ctx, "snapshot_id")
    if snapshot_id is None:
        return fail_missing_required("snapshot_id")
    output_raw = ctx.params.raw.get("output")
    if not isinstance(output_raw, Path):
        return fail_missing_required("output")
    output_path = output_raw
    data_format = ctx.params.get_str("data_format") or _format_from_path(output_path)
    if data_format not in {"ipc", "parquet"}:
        return fail_invalid_value(
            "data_format",
            data_format,
            "data_format must be 'ipc' or 'parquet'",
        )

    try:
        table = _load_table(table_key)
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.time_travel", str(exc))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = table.scan(snapshot_id=snapshot_id).to_arrow_batch_reader()
    if data_format == "parquet":
        row_count = _write_parquet(reader, output_path)
    else:
        row_count = _write_ipc(reader, output_path)

    return CliResult.ok(
        IcebergTimeTravelResult(
            table_key=table_key,
            snapshot_id=snapshot_id,
            output_path=str(output_path),
            data_format=data_format,
            row_count=row_count,
        )
    )


def iceberg_add_files_handler(
    ctx: CommandContext,
) -> CliResult[IcebergAddFilesResult | None]:
    """Add Parquet files to an Iceberg table.

    Returns
    -------
    CliResult[object | None]
        CLI result with add_files payload or error info.
    """
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_missing_required("table_key")
    data_raw = ctx.params.raw.get("data_path")
    if not isinstance(data_raw, Path):
        return fail_missing_required("data_path")
    data_path = data_raw
    if not data_path.exists():
        return fail_invalid_value("data_path", str(data_path), "data path not found")
    refresh_cache = ctx.params.get_bool("refresh_cache")
    if refresh_cache and not ctx.has_storage:
        return fail_project_error(
            "iceberg.add_files",
            "refresh_cache requires storage access (--db-path or project config)",
        )

    try:
        request = IcebergAddFilesRequest(
            table_key=table_key,
            data_dir=data_path if data_path.is_dir() else None,
            file_paths=None if data_path.is_dir() else (data_path,),
            gateway=ctx.gateway if refresh_cache else None,
        )
        result = add_files_to_iceberg(
            request,
            settings=_load_iceberg_settings(),
        )
    except ICEBERG_HANDLER_ERRORS as exc:
        return fail_project_error("iceberg.add_files", str(exc))

    return CliResult.ok(
        IcebergAddFilesResult(
            table_key=result.table_key,
            created=result.created,
            file_count=result.file_count,
            snapshot_id=result.snapshot_id,
        )
    )


def iceberg_refresh_cache_handler(
    ctx: CommandContext,
) -> CliResult[IcebergRefreshCacheResult | None]:
    """Refresh the Iceberg metadata cache.

    Returns
    -------
    CliResult[IcebergRefreshCacheResult | None]
        CLI result with refresh payload or error info.
    """
    if not ctx.has_storage:
        return fail_project_error(
            "iceberg.refresh_cache",
            "refresh_cache requires storage access (--db-path or project config)",
        )
    table_key = ctx.params.get_str("table_key")
    if table_key:
        table_keys = (table_key,)
    else:
        table_keys = tuple(
            key for key, dataset in ctx.gateway.datasets.by_table_key.items() if not dataset.is_view
        )
    if not table_keys:
        return CliResult.ok(IcebergRefreshCacheResult(table_keys=(), refreshed=0, skipped=0))
    settings = _load_iceberg_settings()
    provider = IcebergCatalogProvider(settings)
    refreshed = 0
    skipped = 0
    for key in table_keys:
        try:
            if not provider.table_exists(key):
                skipped += 1
                continue
            table = provider.load_table(key)
        except ICEBERG_HANDLER_ERRORS:
            skipped += 1
            continue
        refresh_iceberg_metadata_cache(
            gateway=ctx.gateway,
            table_key=key,
            table=table,
        )
        refreshed += 1
    return CliResult.ok(
        IcebergRefreshCacheResult(
            table_keys=table_keys,
            refreshed=refreshed,
            skipped=skipped,
        )
    )


def _load_table(table_key: str) -> Table:
    settings = _load_iceberg_settings()
    provider = IcebergCatalogProvider(settings)
    return provider.load_table(table_key)


def _parse_manage_snapshots_request(
    ctx: CommandContext,
) -> _ManageSnapshotsRequest | CliResult[IcebergManageSnapshotsResult | None]:
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_missing_required("table_key")
    ref_name = ctx.params.get_str("ref_name")
    if not ref_name:
        return fail_missing_required("ref_name")
    ref_type_raw = ctx.params.get_str("ref_type")
    ref_type: Literal["tag", "branch"]
    if ref_type_raw == "tag":
        ref_type = "tag"
    elif ref_type_raw == "branch":
        ref_type = "branch"
    else:
        return fail_invalid_value(
            "ref_type",
            ref_type_raw,
            f"ref_type must be one of: {', '.join(_REF_TYPES)}",
        )
    remove_ref = ctx.params.get_bool("ref_remove")
    snapshot_id = _optional_int(ctx, "snapshot_id")
    if remove_ref:
        confirm = ctx.params.get_bool("confirm")
        if not confirm:
            return fail_invalid_value(
                "confirm",
                confirm,
                "confirmation is required to remove refs",
                suggestion="Pass --confirm to remove the ref",
            )
    elif snapshot_id is None:
        return fail_missing_required("snapshot_id")
    return _ManageSnapshotsRequest(
        table_key=table_key,
        ref_name=ref_name,
        ref_type=ref_type,
        remove_ref=remove_ref,
        snapshot_id=snapshot_id,
    )


def _apply_manage_snapshots(table: Table, *, request: _ManageSnapshotsRequest) -> None:
    with table.manage_snapshots() as manager:
        if request.remove_ref:
            if request.ref_type == "tag":
                manager.remove_tag(request.ref_name)
            else:
                manager.remove_branch(request.ref_name)
            return
        snapshot_id = request.snapshot_id
        if snapshot_id is None:
            msg = "snapshot_id is required to create a snapshot ref"
            raise ValueError(msg)
        if request.ref_type == "tag":
            manager.create_tag(snapshot_id, request.ref_name)
        else:
            manager.create_branch(snapshot_id, request.ref_name)


def _load_iceberg_settings() -> IcebergSettings:
    return SettingsView.from_runtime().build.iceberg


def _optional_int(ctx: CommandContext, key: str) -> int | None:
    raw = ctx.params.raw.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return None


def _table_to_rows(table: pa.Table) -> list[dict[str, object]]:
    try:
        return table.to_pylist()
    except (TypeError, ValueError):
        payload = table.to_pydict()
        columns = list(payload.keys())
        row_count = len(next(iter(payload.values()), []))
        rows: list[dict[str, object]] = []
        for row_idx in range(row_count):
            row: dict[str, object] = {}
            for col in columns:
                values = payload.get(col, [])
                row[col] = values[row_idx] if row_idx < len(values) else None
            rows.append(row)
        return rows


def _expired_snapshot_ids(table: Table, *, cutoff: datetime) -> list[int]:
    cutoff_ms = int(cutoff.timestamp() * 1000)
    protected_ids = {
        ref.snapshot_id
        for ref in table.metadata.refs.values()
        if ref.snapshot_ref_type in {SnapshotRefType.BRANCH, SnapshotRefType.TAG}
    }
    expired: list[int] = []
    for snapshot in table.metadata.snapshots:
        if snapshot.snapshot_id in protected_ids:
            continue
        if snapshot.timestamp_ms < cutoff_ms:
            expired.append(snapshot.snapshot_id)
    expired.sort()
    return expired


def _format_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    return "ipc"


def _write_parquet(reader: pa.RecordBatchReader, output_path: Path) -> int:
    row_count = 0
    with pq.ParquetWriter(str(output_path), reader.schema) as writer:
        for batch in reader:
            writer.write_batch(batch)
            row_count += batch.num_rows
    return row_count


def _write_ipc(reader: pa.RecordBatchReader, output_path: Path) -> int:
    row_count = 0
    with (
        pa.OSFile(str(output_path), "wb") as sink,
        pa.ipc.new_file(sink, reader.schema) as writer,
    ):
        for batch in reader:
            writer.write_batch(batch)
            row_count += batch.num_rows
    return row_count


__all__ = [
    "iceberg_add_files_handler",
    "iceberg_expire_snapshots_handler",
    "iceberg_inspect_handler",
    "iceberg_manage_snapshots_handler",
    "iceberg_refresh_cache_handler",
    "iceberg_refs_handler",
    "iceberg_time_travel_handler",
]
