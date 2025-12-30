"""Iceberg metadata-derived statistics helpers."""

from __future__ import annotations

from typing import Protocol

from codeintel.storage.tracking.schema_catalog_models import IcebergStatsPayload

_SUMMARY_TOTAL_RECORDS = ("total-records", "total_records")
_SUMMARY_DATA_FILES = ("total-data-files", "total_data_files")
_SUMMARY_DELETE_FILES = ("total-delete-files", "total_delete_files")
_SUMMARY_TOTAL_BYTES = ("total-files-size", "total_files_size", "total-file-size")


class _ManifestTable(Protocol):
    @property
    def num_rows(self) -> int: ...


class _InspectTable(Protocol):
    def manifests(self) -> _ManifestTable: ...


class _IcebergTable(Protocol):
    def snapshot_by_id(self, snapshot_id: int) -> object | None: ...

    def current_snapshot(self) -> object | None: ...

    @property
    def inspect(self) -> _InspectTable: ...


def iceberg_stats_for_table(
    table: _IcebergTable,
    *,
    snapshot_id: int | None = None,
    include_manifests: bool = True,
) -> IcebergStatsPayload:
    """Return Iceberg stats derived from table metadata.

    Parameters
    ----------
    table
        Iceberg table to inspect.
    snapshot_id
        Optional snapshot id to target. Defaults to current snapshot.
    include_manifests
        Whether to compute manifest_count via inspect.manifests().

    Returns
    -------
    IcebergStatsPayload
        Metadata-derived stats payload for the snapshot, or empty if unavailable.
    """
    snapshot_obj = (
        table.snapshot_by_id(snapshot_id) if snapshot_id is not None else table.current_snapshot()
    )
    if snapshot_obj is None:
        return {}
    payload_and_summary = _snapshot_payload(snapshot_obj)
    if payload_and_summary is None:
        return {}
    payload, summary = payload_and_summary
    _apply_snapshot_count(payload, table)
    _apply_summary_metrics(payload, summary)
    if include_manifests:
        _apply_manifest_count(payload, table)
    return payload


def _summary_int(summary: dict[str, str], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        raw = summary.get(key)
        if raw is None:
            continue
        value = _coerce_int(raw)
        if value is not None:
            return value
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
    return None


def _manifest_count(table: _IcebergTable) -> int | None:
    try:
        manifest_table = table.inspect.manifests()
    except (RuntimeError, ValueError, TypeError, OSError):
        return None
    try:
        return int(manifest_table.num_rows)
    except (TypeError, ValueError, AttributeError):
        return None


def _snapshot_count(table: _IcebergTable) -> int | None:
    metadata = getattr(table, "metadata", None)
    snapshots = getattr(metadata, "snapshots", None) if metadata is not None else None
    if snapshots is None:
        return None
    try:
        return len(snapshots)
    except TypeError:
        return None


def _snapshot_payload(
    snapshot_obj: object,
) -> tuple[IcebergStatsPayload, dict[str, str]] | None:
    snapshot_id_value = getattr(snapshot_obj, "snapshot_id", None)
    if not isinstance(snapshot_id_value, int):
        return None
    payload: IcebergStatsPayload = {"snapshot_id": snapshot_id_value}
    schema_id_value = getattr(snapshot_obj, "schema_id", None)
    if isinstance(schema_id_value, int):
        payload["schema_id"] = schema_id_value
    summary_obj = getattr(snapshot_obj, "summary", None)
    summary_props = getattr(summary_obj, "additional_properties", None)
    summary = summary_props if isinstance(summary_props, dict) else {}
    return payload, summary


def _apply_snapshot_count(payload: IcebergStatsPayload, table: _IcebergTable) -> None:
    snapshot_count = _snapshot_count(table)
    if snapshot_count is not None:
        payload["snapshot_count"] = snapshot_count


def _apply_summary_metrics(payload: IcebergStatsPayload, summary: dict[str, str]) -> None:
    total_records = _summary_int(summary, _SUMMARY_TOTAL_RECORDS)
    if total_records is not None:
        payload["total_records"] = total_records
    data_files = _summary_int(summary, _SUMMARY_DATA_FILES)
    if data_files is not None:
        payload["data_file_count"] = data_files
    delete_files = _summary_int(summary, _SUMMARY_DELETE_FILES)
    if delete_files is not None:
        payload["delete_file_count"] = delete_files
    total_bytes = _summary_int(summary, _SUMMARY_TOTAL_BYTES)
    if total_bytes is not None:
        payload["total_bytes"] = total_bytes


def _apply_manifest_count(payload: IcebergStatsPayload, table: _IcebergTable) -> None:
    manifest_count = _manifest_count(table)
    if manifest_count is not None:
        payload["manifest_count"] = manifest_count


__all__ = ["iceberg_stats_for_table"]
