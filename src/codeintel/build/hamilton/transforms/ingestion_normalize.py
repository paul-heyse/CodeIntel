"""Normalization utilities for ingestion Arrow outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypedDict, Unpack

import pyarrow as pa

from codeintel.build.schemas.registry import require_table_schema
from codeintel.build.tabular.arrow_ops import (
    AlignmentReport,
    AlignmentReporter,
    align_table_to_contract,
    dedupe_table_for_table,
    emit_alignment_report,
)
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.finalize_ops import (
    FinalizeDedupe,
    FinalizeInvariant,
    FinalizeMode,
    FinalizeResult,
    FinalizeSpec,
    finalize_reader,
    finalize_table,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.validation.schema_constraints import list_alignment_specs_for_table_key


class NormalizationOverrides(TypedDict, total=False):
    """Keyword overrides for ingestion normalization."""

    target_name: str | None
    add_missing: bool
    keep_extras: bool | None
    reporter: AlignmentReporter | None


@dataclass(frozen=True, slots=True)
class NormalizationOptions:
    """Options for ingestion normalization."""

    target_name: str | None = None
    add_missing: bool = True
    keep_extras: bool | None = None
    reporter: AlignmentReporter | None = None


_TOLERANT_INGEST_TABLE_KEYS = frozenset(
    {
        "core.scip_diagnostics",
        "core.scip_index_metadata",
        "core.ts_changed_ranges",
        "core.ts_parse_errors",
        "core.ts_weld_coverage",
    }
)

_DEDUPE_PREFER_COLUMNS: dict[str, Sequence[str]] = {
    "core.file_state": ("mtime_ns", "content_hash"),
    "core.scip_external_symbols": ("package_manager", "package_name", "package_version"),
}


def _merge_normalization_options(
    options: NormalizationOptions | None,
    overrides: NormalizationOverrides,
) -> NormalizationOptions:
    resolved = options or NormalizationOptions()
    if overrides:
        return replace(resolved, **overrides)
    return resolved


def _required_non_null_columns(table_key: str) -> tuple[str, ...]:
    schema = require_table_schema(table_key)
    return tuple(column.name for column in schema.columns if not column.nullable)


def _key_fields_for_table(table_key: str) -> tuple[str, ...]:
    schema = require_table_schema(table_key)
    return tuple(schema.primary_key) if schema.primary_key else ()


def _list_alignment_invariants(table_key: str) -> tuple[FinalizeInvariant, ...]:
    specs = list_alignment_specs_for_table_key(table_key)
    return tuple(FinalizeInvariant.list_alignment(spec.column, spec.related) for spec in specs)


def _finalize_mode_for_table(table_key: str) -> FinalizeMode:
    return "tolerant" if table_key in _TOLERANT_INGEST_TABLE_KEYS else "strict"


def _finalize_dedupe(table_key: str) -> FinalizeDedupe | None:
    prefer = _DEDUPE_PREFER_COLUMNS.get(table_key)
    if prefer is None:
        return None
    return FinalizeDedupe(prefer_columns=prefer)


def _emit_alignment_report_from_finalize(result: FinalizeResult) -> None:
    if result.alignment.num_rows == 0:
        return
    row = next(iter_rows(result.alignment), None)
    if row is None:
        return
    table_key_value = row.get("table_key")
    target_name_value = row.get("target_name")
    row_count_value = row.get("row_count")
    report = AlignmentReport(
        table_key=table_key_value if isinstance(table_key_value, str) else "",
        target_name=target_name_value if isinstance(target_name_value, str) else None,
        missing_columns=_string_list(row.get("missing_columns")),
        extra_columns=_string_list(row.get("extra_columns")),
        coerced_columns=_string_list(row.get("coerced_columns")),
        row_count=row_count_value if isinstance(row_count_value, int) else None,
    )
    emit_alignment_report(report)


def _string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _ingest_finalize_spec(
    table_key: str,
    *,
    target_name: str | None,
    mode: FinalizeMode | None,
) -> FinalizeSpec:
    return FinalizeSpec(
        table_key=table_key,
        mode=mode or _finalize_mode_for_table(table_key),
        required_non_null=_required_non_null_columns(table_key),
        invariants=_list_alignment_invariants(table_key),
        key_fields=_key_fields_for_table(table_key),
        dedupe=_finalize_dedupe(table_key),
        emit_artifacts=True,
        target_name=target_name,
    )


def finalize_ingest_table(
    table_key: str,
    table: pa.Table,
    *,
    target_name: str | None,
    mode: FinalizeMode | None = None,
) -> pa.Table:
    """Finalize an ingestion table using the shared policy table.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    table
        Arrow table to finalize.
    target_name
        Target name used for finalize artifacts.
    mode
        Optional override for the finalize mode.

    Returns
    -------
    pa.Table
        Finalized table containing valid rows.
    """
    spec = _ingest_finalize_spec(table_key, target_name=target_name, mode=mode)
    result = finalize_table(table, spec=spec)
    _emit_alignment_report_from_finalize(result)
    return result.good


def finalize_ingest_reader(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    target_name: str | None,
    mode: FinalizeMode | None = None,
) -> pa.Table:
    """Finalize an ingestion reader using the shared policy table.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    reader
        Arrow record batch reader to finalize.
    target_name
        Target name used for finalize artifacts.
    mode
        Optional override for the finalize mode.

    Returns
    -------
    pa.Table
        Finalized table containing valid rows.
    """
    spec = _ingest_finalize_spec(table_key, target_name=target_name, mode=mode)
    result = finalize_reader(reader, spec=spec)
    _emit_alignment_report_from_finalize(result)
    return result.good


def normalize_ingest_frame(
    frame: InferableTabularInput | None,
    *,
    table_key: str,
    options: NormalizationOptions | None = None,
    **overrides: Unpack[NormalizationOverrides],
) -> pa.Table | None:
    """Normalize ingestion frames for schema alignment and deduping.

    Parameters
    ----------
    frame
        Tabular input to normalize (None means skip).
    table_key
        Target table key for schema alignment.
    options
        Optional normalization options for alignment and deduping.
    overrides
        Keyword overrides for normalization options.

    Returns
    -------
    pa.Table | None
        Normalized Arrow table or None if input is None.
    """
    if frame is None:
        return None
    table = tabular_to_arrow_table(frame)
    if table.num_rows == 0:
        return pa.Table.from_batches([], schema=table.schema)
    resolved_options = _merge_normalization_options(options, overrides)
    extras_policy = None
    if resolved_options.keep_extras is True:
        extras_policy = "retain"
    elif resolved_options.keep_extras is False:
        extras_policy = "drop"
    aligned = (
        align_table_to_contract(
            table_key,
            table,
            target_name=resolved_options.target_name,
            extras_policy=extras_policy,
            reporter=resolved_options.reporter,
        )
        if resolved_options.add_missing or extras_policy is not None
        else table
    )
    return dedupe_table_for_table(table_key, aligned)


__all__ = ["finalize_ingest_reader", "finalize_ingest_table", "normalize_ingest_frame"]
