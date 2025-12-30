"""Encode and decode schema observation payloads."""

from __future__ import annotations

import base64
from collections.abc import Mapping, MutableMapping
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal, Protocol, TypeGuard, cast

from codeintel.core.schemas.contracts import ExtrasPolicy
from codeintel.core.schemas.primitives import TableSchema
from codeintel.storage.helpers.json import decode_json_dict
from codeintel.storage.tracking.schema_catalog_models import (
    ColumnStatsEntry,
    ColumnStatsPayload,
    DatasetStatsPayload,
    DerivedSettingsPayload,
    ParquetStatsPayload,
)


class ColumnStatsLike(Protocol):
    """Protocol for column stats accumulators used by the encoder."""

    @property
    def null_count(self) -> int:
        """Return the null count."""
        ...

    @property
    def non_null_count(self) -> int:
        """Return the non-null count."""
        ...

    @property
    def distinct_max(self) -> int | None:
        """Return the max distinct count."""
        ...

    @property
    def min_value(self) -> object | None:
        """Return the minimum observed value."""
        ...

    @property
    def max_value(self) -> object | None:
        """Return the maximum observed value."""
        ...

    @property
    def length_sum(self) -> int:
        """Return the sum of observed lengths."""
        ...

    @property
    def length_count(self) -> int:
        """Return the count of observed lengths."""
        ...


_DEFAULT_DICT_MAX_CARDINALITY = 256
_DEFAULT_DICT_RATIO = 0.1
_TARGET_ROW_GROUP_BYTES = 64 * 1024 * 1024
_MIN_ROW_GROUP_SIZE = 10_000
_MAX_ROW_GROUP_SIZE = 1_000_000
_MIN_DATA_PAGE_SIZE = 64 * 1024
_MAX_DATA_PAGE_SIZE = 1024 * 1024
_ROW_GROUP_PAGE_DIVISOR = 128


def encode_derived_settings(
    *,
    table_schema: TableSchema,
    column_stats: Mapping[str, ColumnStatsLike],
    row_count: int,
    total_bytes: int,
    extras_policy: ExtrasPolicy,
) -> DerivedSettingsPayload | None:
    """Compute derived settings payload from observed stats.

    Returns
    -------
    DerivedSettingsPayload | None
        Derived settings payload when any settings are inferred, otherwise None.
    """
    settings: DerivedSettingsPayload = {"extras_policy": extras_policy}
    dictionary_columns: list[str] = []
    distinct_values: list[int] = []
    for column in table_schema.columns:
        if column.type != "VARCHAR":
            continue
        stats = column_stats.get(column.name)
        if stats is None or stats.distinct_max is None:
            continue
        if stats.non_null_count <= 0:
            continue
        distinct = stats.distinct_max
        ratio = distinct / stats.non_null_count
        if distinct <= _DEFAULT_DICT_MAX_CARDINALITY and ratio <= _DEFAULT_DICT_RATIO:
            dictionary_columns.append(column.name)
            distinct_values.append(distinct)
    if dictionary_columns:
        settings["dictionary_encode_columns"] = sorted(dictionary_columns)
        settings["dictionary_max_cardinality"] = max(distinct_values)
        settings["unify_dictionaries"] = True

    if row_count > 0 and total_bytes > 0:
        avg_row_bytes = total_bytes / row_count
        if avg_row_bytes > 0:
            raw_rows = int(_TARGET_ROW_GROUP_BYTES / avg_row_bytes)
            row_group_size = max(_MIN_ROW_GROUP_SIZE, min(_MAX_ROW_GROUP_SIZE, raw_rows))
            row_group_bytes = row_group_size * avg_row_bytes
            page_bytes = int(row_group_bytes / _ROW_GROUP_PAGE_DIVISOR)
            page_bytes = max(_MIN_DATA_PAGE_SIZE, min(_MAX_DATA_PAGE_SIZE, page_bytes))
            settings["row_group_size"] = row_group_size
            settings["data_page_size"] = page_bytes
            settings["avg_row_bytes"] = avg_row_bytes

    return settings or None


def encode_column_stats(
    column_stats: Mapping[str, ColumnStatsLike],
) -> ColumnStatsPayload | None:
    """Encode column stats accumulators into a payload.

    Returns
    -------
    ColumnStatsPayload | None
        Encoded column stats payload, or None when no stats are provided.
    """
    if not column_stats:
        return None
    payload: ColumnStatsPayload = {}
    for name, stats in column_stats.items():
        entry: ColumnStatsEntry = {
            "null_count": stats.null_count,
            "non_null_count": stats.non_null_count,
        }
        if stats.distinct_max is not None:
            entry["distinct_count_max"] = stats.distinct_max
        if stats.min_value is not None:
            entry["min"] = _json_safe_value(stats.min_value)
        if stats.max_value is not None:
            entry["max"] = _json_safe_value(stats.max_value)
        if stats.length_count > 0:
            entry["avg_length"] = stats.length_sum / stats.length_count
        payload[name] = entry
    return payload


def encode_dataset_stats(
    *,
    row_count: int,
    batch_count: int,
    total_bytes: int,
    manifest_stats: ParquetStatsPayload | None,
    manifest_row_count: int | None,
) -> DatasetStatsPayload:
    """Encode dataset statistics into a payload.

    Returns
    -------
    DatasetStatsPayload
        Encoded dataset stats payload.
    """
    payload: DatasetStatsPayload = {
        "row_count": row_count,
        "batch_count": batch_count,
        "total_bytes": total_bytes,
    }
    if manifest_row_count is not None:
        payload["manifest_row_count"] = manifest_row_count
    if manifest_stats:
        payload["parquet_stats"] = dict(manifest_stats)
    return payload


def _json_safe_value(value: object) -> object:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(value).decode("ascii")
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def decode_column_stats(value: object | None) -> ColumnStatsPayload | None:
    """Decode column stats payloads from stored JSON values.

    Returns
    -------
    ColumnStatsPayload | None
        Decoded column stats payload, or None when invalid.
    """
    decoded = _decode_optional_json_dict(value)
    if decoded is None:
        return None
    payload: ColumnStatsPayload = {}
    for column_name, entry_raw in decoded.items():
        if not isinstance(column_name, str):
            return None
        entry = _coerce_column_stats_entry(entry_raw)
        if entry is None:
            return None
        payload[column_name] = entry
    return payload or None


def _coerce_column_stats_entry(value: object) -> ColumnStatsEntry | None:
    if not isinstance(value, Mapping):
        return None
    entry: ColumnStatsEntry = {}
    for key in ("null_count", "non_null_count", "distinct_count_max"):
        raw = value.get(key)
        if raw is None:
            continue
        if not _is_int(raw):
            return None
        entry[key] = raw
    for key in ("avg_length",):
        raw = value.get(key)
        if raw is None:
            continue
        if not _is_floatlike(raw):
            return None
        entry[key] = float(raw)
    for key in ("min", "max"):
        if key in value:
            entry[key] = value[key]
    return entry


type _DatasetStatsKey = Literal[
    "row_count",
    "batch_count",
    "total_bytes",
    "manifest_row_count",
]


_DATASET_INT_KEYS: tuple[_DatasetStatsKey, ...] = (
    "row_count",
    "batch_count",
    "total_bytes",
    "manifest_row_count",
)


def decode_dataset_stats(value: object | None) -> DatasetStatsPayload | None:
    """Decode dataset stats payloads from stored JSON values.

    Returns
    -------
    DatasetStatsPayload | None
        Decoded dataset stats payload, or None when invalid.
    """
    decoded = _decode_optional_json_dict(value)
    if decoded is None:
        return None
    payload: DatasetStatsPayload = {}
    for key in _DATASET_INT_KEYS:
        if not _apply_optional_dataset_int(payload, decoded, key):
            return None
    parquet_stats = decoded.get("parquet_stats")
    if parquet_stats is not None:
        parquet_payload = _coerce_string_object_mapping(parquet_stats)
        if parquet_payload is None:
            return None
        payload["parquet_stats"] = parquet_payload
    return payload or None


type _DerivedIntKey = Literal[
    "dictionary_max_cardinality",
    "row_group_size",
    "data_page_size",
]


_DERIVED_INT_KEYS: tuple[_DerivedIntKey, ...] = (
    "dictionary_max_cardinality",
    "row_group_size",
    "data_page_size",
)


def decode_derived_settings(value: object | None) -> DerivedSettingsPayload | None:
    """Decode derived settings payloads from stored JSON values.

    Returns
    -------
    DerivedSettingsPayload | None
        Decoded derived settings payload, or None when invalid.
    """
    decoded = _decode_optional_json_dict(value)
    if decoded is None:
        return None
    payload: dict[str, object] = {}
    valid = _apply_optional_str(payload, decoded, "extras_policy") and _apply_optional_str_list(
        payload, decoded, "dictionary_encode_columns"
    )
    if valid:
        for key in _DERIVED_INT_KEYS:
            if not _apply_optional_derived_int(payload, decoded, key):
                valid = False
                break
    if valid:
        valid = _apply_optional_bool(
            payload, decoded, "unify_dictionaries"
        ) and _apply_optional_float(payload, decoded, "avg_row_bytes")
    if not valid:
        return None
    if not payload:
        return None
    return cast("DerivedSettingsPayload", payload)


def _decode_optional_json_dict(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    decoded = decode_json_dict(value)
    return decoded if decoded else None


def _coerce_string_object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    payload: dict[str, object] = {}
    for key, entry in value.items():
        if not isinstance(key, str):
            return None
        payload[key] = entry
    return payload


def _apply_optional_dataset_int(
    payload: DatasetStatsPayload,
    decoded: Mapping[str, object],
    key: _DatasetStatsKey,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not _is_int(raw):
        return False
    payload[key] = raw
    return True


def _apply_optional_str(
    payload: MutableMapping[str, object],
    decoded: Mapping[str, object],
    key: str,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not isinstance(raw, str):
        return False
    payload[key] = raw
    return True


def _apply_optional_str_list(
    payload: MutableMapping[str, object],
    decoded: Mapping[str, object],
    key: str,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not isinstance(raw, list):
        return False
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            return False
        values.append(item)
    payload[key] = values
    return True


def _apply_optional_derived_int(
    payload: MutableMapping[str, object],
    decoded: Mapping[str, object],
    key: _DerivedIntKey,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not _is_int(raw):
        return False
    payload[key] = raw
    return True


def _apply_optional_bool(
    payload: MutableMapping[str, object],
    decoded: Mapping[str, object],
    key: str,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not isinstance(raw, bool):
        return False
    payload[key] = raw
    return True


def _apply_optional_float(
    payload: MutableMapping[str, object],
    decoded: Mapping[str, object],
    key: str,
) -> bool:
    raw = decoded.get(key)
    if raw is None:
        return True
    if not _is_floatlike(raw):
        return False
    payload[key] = float(raw)
    return True


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_floatlike(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "ColumnStatsLike",
    "decode_column_stats",
    "decode_dataset_stats",
    "decode_derived_settings",
    "encode_column_stats",
    "encode_dataset_stats",
    "encode_derived_settings",
]
