"""Normalization utilities for ingestion Arrow outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack

import pyarrow as pa

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import (
    AlignmentReport,
    AlignmentReporter,
    align_table_to_contract,
    dedupe_table_for_table,
    emit_alignment_report,
)
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.finalize_ops import (
    FinalizeMode,
    FinalizeResult,
    finalize_spec_for_table,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import ExecutionContext, resolve_execution_context
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.columnar.streaming import ScanTelemetry
from codeintel.ingestion.compute.plan_surface import IngestQuery, ingest_plan_for_table

if TYPE_CHECKING:
    from codeintel.build.tabular.finalize_ops import FinalizeSpec


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


@dataclass(frozen=True, slots=True)
class IngestFinalizeOptions:
    """Options for ingestion finalization."""

    target_name: str | None = None
    mode: FinalizeMode | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None
    scan_telemetry: ScanTelemetry | None = None


_TOLERANT_INGEST_TABLE_KEYS = frozenset(
    {
        "core.scip_diagnostics",
        "core.scip_index_metadata",
        "core.ts_changed_ranges",
        "core.ts_parse_errors",
        "core.ts_weld_coverage",
    }
)


def _merge_normalization_options(
    options: NormalizationOptions | None,
    overrides: NormalizationOverrides,
) -> NormalizationOptions:
    resolved = options or NormalizationOptions()
    if overrides:
        return replace(resolved, **overrides)
    return resolved


def _finalize_mode_for_table(table_key: str) -> FinalizeMode:
    return "tolerant" if table_key in _TOLERANT_INGEST_TABLE_KEYS else "strict"


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
    return finalize_spec_for_table(
        table_key,
        mode=mode or _finalize_mode_for_table(table_key),
        emit_artifacts=True,
        target_name=target_name,
    )


def finalize_ingest_table(
    table_key: str,
    table: pa.Table,
    *,
    options: IngestFinalizeOptions | None = None,
) -> pa.Table:
    """Finalize an ingestion table using the shared policy table.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    table
        Arrow table to finalize.
    options
        Optional finalize options for target name, mode, and run manifests.

    Returns
    -------
    pa.Table
        Finalized table containing valid rows.
    """
    resolved = options or IngestFinalizeOptions()
    spec = _ingest_finalize_spec(
        table_key,
        target_name=resolved.target_name,
        mode=resolved.mode,
    )
    resolved_ctx = resolve_execution_context(None)
    plan = ExecutionPlan.from_table(
        table,
        ordering=OrderingSpec.implicit(reason="ingest table"),
    )
    run_options = _pipeline_run_options(
        table_key=table_key,
        ctx=resolved_ctx,
        options=resolved,
    )
    result = run_pipeline(plan=plan, finalize=spec, options=run_options)
    _emit_alignment_report_from_finalize(result)
    return result.good


def finalize_ingest_reader(
    table_key: str,
    reader: pa.RecordBatchReader,
    *,
    options: IngestFinalizeOptions | None = None,
) -> pa.Table:
    """Finalize an ingestion reader using the shared policy table.

    Parameters
    ----------
    table_key
        Fully qualified table key.
    reader
        Arrow record batch reader to finalize.
    options
        Optional finalize options for target name, mode, and run manifests.

    Returns
    -------
    pa.Table
        Finalized table containing valid rows.
    """
    resolved = options or IngestFinalizeOptions()
    spec = _ingest_finalize_spec(
        table_key,
        target_name=resolved.target_name,
        mode=resolved.mode,
    )
    resolved_ctx = resolve_execution_context(None)
    plan = ExecutionPlan.from_reader(
        reader,
        ordering=OrderingSpec.implicit(reason="ingest reader"),
    )
    run_options = _pipeline_run_options(
        table_key=table_key,
        ctx=resolved_ctx,
        options=resolved,
    )
    result = run_pipeline(plan=plan, finalize=spec, options=run_options)
    _emit_alignment_report_from_finalize(result)
    return result.good


def _pipeline_run_options(
    *,
    table_key: str,
    ctx: ExecutionContext,
    options: IngestFinalizeOptions,
) -> PipelineRunOptions:
    if options.manifest_dir is None:
        return PipelineRunOptions(ctx=ctx)
    manifest_options = options.manifest_options or RunManifestOptions(
        filename=f"run_manifest_{table_key.replace('.', '_')}.json",
        extras=_manifest_extras(
            table_key=table_key,
            target_name=options.target_name,
        ),
    )
    return PipelineRunOptions(
        ctx=ctx,
        manifest_dir=options.manifest_dir,
        manifest_options=manifest_options,
        scan_telemetry=options.scan_telemetry,
    )


def _manifest_extras(
    *,
    table_key: str,
    target_name: str | None,
) -> dict[str, object]:
    extras: dict[str, object] = {"table_key": table_key}
    if target_name is not None:
        extras["target_name"] = target_name
    return extras


def scoped_table_for_ingest(
    value: InferableTabularInput,
    *,
    table_key: str,
    scope: SnapshotScope | None,
    columns: Sequence[str] | None,
    require_scope_columns: bool,
) -> pa.Table:
    """Return a scope-filtered, projected table using QuerySpec plan helpers.

    Parameters
    ----------
    value
        Tabular input to scope and project.
    table_key
        Table key used to resolve projection defaults.
    scope
        Optional snapshot scope (repo/commit) for filtering.
    columns
        Optional columns to project (None keeps all).
    require_scope_columns
        Whether missing repo/commit columns should raise.

    Returns
    -------
    pyarrow.Table
        Scoped and projected table.

    Raises
    ------
    ValueError
        If required scope columns are missing from the table.
    """
    table = tabular_to_arrow_table(value)
    if table.num_rows == 0:
        return table.select(list(columns)) if columns is not None else table
    if scope is None:
        return table.select(list(columns)) if columns is not None else table
    missing = [name for name in ("repo", "commit") if name not in table.column_names]
    if missing:
        if require_scope_columns:
            msg = f"Missing snapshot columns: {missing}"
            raise ValueError(msg)
        return table.select(list(columns)) if columns is not None else table
    projection_columns = tuple(columns) if columns is not None else None
    query = IngestQuery(
        table_key=table_key,
        columns=projection_columns,
        repo=scope.repo,
        commit=scope.commit,
    )
    resolved_ctx = resolve_execution_context(None)
    plan = ingest_plan_for_table(table, query=query, ctx=resolved_ctx)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=resolved_ctx)
    return reader_to_table(reader)


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


__all__ = [
    "IngestFinalizeOptions",
    "finalize_ingest_reader",
    "finalize_ingest_table",
    "normalize_ingest_frame",
    "scoped_table_for_ingest",
]
