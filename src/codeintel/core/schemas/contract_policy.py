"""Shared contract policy helpers for export defaults and schema IDs."""

from __future__ import annotations

from typing import Literal

from codeintel.core.schemas.primitives import TableSchema
from codeintel.storage.helpers.table_key import try_parse_table_key

__all__ = [
    "default_json_schema_id",
    "default_jsonl_filename",
    "default_parquet_filename",
    "exportable_by_default",
    "table_name_from_key",
]

_NON_EXPORTABLE_CORE_TABLES: frozenset[str] = frozenset(
    {
        "file_state",
        "ingest_runs",
        "repo_map",
        "schema_inference_errors",
        "scip_occurrences",
        "scip_symbols",
        "test_results",
        "test_summary",
    }
)

_NON_EXPORTABLE_ANALYTICS_TABLES: frozenset[str] = frozenset({"tags_index"})


def table_name_from_key(table_key: str) -> str:
    """Return the dataset/table name component of a table key.

    Returns
    -------
    str
        Table name portion of the key.
    """
    parsed = try_parse_table_key(table_key)
    if parsed is None:
        return table_key
    return parsed.name


def exportable_by_default(table_key: str) -> bool:
    """Return True when a dataset should be exported by default.

    Returns
    -------
    bool
        True when the dataset is exportable by default.
    """
    parsed = try_parse_table_key(table_key)
    if parsed is None:
        return False
    schema_prefix = parsed.schema
    table_name = parsed.name

    if schema_prefix == "build":
        return False

    if schema_prefix == "core":
        return table_name not in _NON_EXPORTABLE_CORE_TABLES

    if schema_prefix == "graph":
        return not (table_name == "import_modules" or table_name.startswith("v_"))

    if schema_prefix == "analytics":
        is_internal_metrics_ext = table_name.endswith("_metrics_ext") and table_name.startswith(
            ("cfg_", "dfg_")
        )
        return (
            table_name not in _NON_EXPORTABLE_ANALYTICS_TABLES
            and not table_name.endswith("_cache")
            and not is_internal_metrics_ext
        )

    return True


def _default_export_filename(
    table_key: str,
    *,
    kind: Literal["jsonl", "parquet"],
) -> str:
    return f"{table_name_from_key(table_key)}.{kind}"


def default_json_schema_id(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic JSON Schema ID for a dataset.

    Returns
    -------
    str | None
        JSON Schema ID for the dataset, or None when unavailable.
    """
    if schema is None:
        return None
    parsed = try_parse_table_key(table_key)
    if parsed is None:
        return None
    schema_prefix = parsed.schema
    table_name = parsed.name
    if schema_prefix == "build":
        return None
    return table_name


def default_jsonl_filename(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic JSONL filename for a dataset.

    Returns
    -------
    str | None
        JSONL filename for the dataset, or None when unavailable.
    """
    if schema is None:
        return None
    if not exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="jsonl")


def default_parquet_filename(*, table_key: str, schema: TableSchema | None) -> str | None:
    """Return deterministic Parquet filename for a dataset.

    Returns
    -------
    str | None
        Parquet filename for the dataset, or None when unavailable.
    """
    if schema is None:
        return None
    if not exportable_by_default(table_key):
        return None
    return _default_export_filename(table_key, kind="parquet")
