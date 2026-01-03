"""Resolved syntax fact tables welded with SCIP occurrences."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import polars as pl
import pyarrow as pa
from polars.exceptions import PolarsError

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import (
    align_reader_to_contract,
    arrow_join_lazyframes,
)
from codeintel.build.tabular.conversion import lazyframe_to_reader, tabular_to_lazyframe
from codeintel.build.tabular.frames import (
    JoinSpec,
    join_validated,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_reader_for_table

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SYNTAX_ENRICH_TARGET_NAME = "syntax_enrich"
SYNTAX_DEFS_RESOLVED_TABLE_KEY = "core.syntax_defs_resolved"
SYNTAX_REFS_RESOLVED_TABLE_KEY = "core.syntax_refs_resolved"
SYNTAX_CALLS_RESOLVED_TABLE_KEY = "core.syntax_calls_resolved"
SYNTAX_IMPORTS_RESOLVED_TABLE_KEY = "core.syntax_imports_resolved"
_OCCURRENCE_INT_COLUMNS = (
    "occ_start_line",
    "occ_start_col",
    "occ_end_line",
    "occ_end_col",
    "occ_start_byte",
    "occ_end_byte",
)


def _ordered_columns(table_key: str) -> list[str]:
    try:
        schema = get_schema_service().require_table_schema(table_key)
    except (KeyError, RuntimeError):
        return []
    return list(schema.column_names())


def _select_for_table(frame: pl.LazyFrame, table_key: str) -> pl.LazyFrame:
    columns = _ordered_columns(table_key)
    if not columns:
        return frame
    existing = set(frame.collect_schema().names())
    missing = [name for name in columns if name not in existing]
    if missing:
        frame = frame.with_columns([pl.lit(None).alias(name) for name in missing])
    return frame.select(columns)


def _merge_column_names(schemas: list[list[str]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for names in schemas:
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            ordered.append(name)
    return ordered


def _align_frames_for_concat(frames: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
    schemas: list[list[str]] = []
    for frame in frames:
        try:
            schemas.append(list(frame.collect_schema().names()))
        except PolarsError:
            return frames
    all_columns = _merge_column_names(schemas)
    aligned: list[pl.LazyFrame] = []
    for frame, names in zip(frames, schemas, strict=True):
        missing = [name for name in all_columns if name not in names]
        resolved = frame
        if missing:
            resolved = frame.with_columns([pl.lit(None).alias(name) for name in missing])
        aligned.append(resolved.select(all_columns))
    return aligned


def _dedupe_for_table(frame: pl.LazyFrame, *, table_key: str) -> pl.LazyFrame:
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return frame
    try:
        columns = set(frame.collect_schema().names())
    except PolarsError:
        return frame
    key_columns = [name for name in schema.primary_key if name in columns]
    if not key_columns:
        return frame
    return frame.unique(subset=key_columns, keep="first", maintain_order=True)


def _occurrence_resolution_frame(
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
) -> pl.LazyFrame:
    def _coerce_occurrence_ints(frame: pl.LazyFrame) -> pl.LazyFrame:
        try:
            columns = set(frame.collect_schema().names())
        except PolarsError:
            return frame
        casts = [
            pl.col(name).cast(pl.Int64, strict=False).alias(name)
            for name in _OCCURRENCE_INT_COLUMNS
            if name in columns
        ]
        if not casts:
            return frame
        return frame.with_columns(casts)

    def _drop_occurrence_bytes(frame: pl.LazyFrame) -> pl.LazyFrame:
        try:
            columns = set(frame.collect_schema().names())
        except PolarsError:
            return frame
        drop_columns = [name for name in ("occ_start_byte", "occ_end_byte") if name in columns]
        if not drop_columns:
            return frame
        return frame.drop(drop_columns)

    span = tabular_to_lazyframe(q__core__scip_occurrence_span_xref).select(
        "repo",
        "commit",
        "rel_path",
        "scip_symbol",
        pl.col("roles").alias("scip_roles"),
        "is_definition",
        "is_reference",
        "is_import",
        "is_write",
        "is_read",
        "goid_h128",
        pl.col("start_line").alias("occ_start_line"),
        pl.col("start_col").alias("occ_start_col"),
        pl.col("end_line").alias("occ_end_line"),
        pl.col("end_col").alias("occ_end_col"),
        pl.col("start_byte").alias("occ_start_byte"),
        pl.col("end_byte").alias("occ_end_byte"),
    )
    span = _coerce_occurrence_ints(span)
    span = _drop_occurrence_bytes(span)
    syntax = tabular_to_lazyframe(q__core__scip_occurrence_syntax_xref).select(
        "repo",
        "commit",
        "rel_path",
        "producer",
        "scip_symbol",
        "scip_occurrence_id",
        "occ_start_byte",
        "occ_end_byte",
        "occ_start_line",
        "occ_start_col",
        "occ_end_line",
        "occ_end_col",
        "syntax_node_id",
        "match_kind",
        "candidate_count",
    )
    syntax = _coerce_occurrence_ints(syntax)
    join_keys = [
        "repo",
        "commit",
        "rel_path",
        "scip_symbol",
        "occ_start_line",
        "occ_start_col",
        "occ_end_line",
        "occ_end_col",
    ]
    # Contract: span rows are unique per occurrence join key.
    return arrow_join_lazyframes(
        syntax,
        span,
        spec=JoinSpec(on=join_keys, how="left", validate="m:1"),
    )


def _resolve_facts(
    facts: pl.LazyFrame,
    occurrences: pl.LazyFrame,
    *,
    table_key: str,
) -> pa.RecordBatchReader:
    fact_columns = list(facts.collect_schema().names())
    if not fact_columns:
        return empty_reader_for_table(table_key)
    resolved_columns = _ordered_columns(table_key)
    matched_bytes, fallback_join, line_join = _resolve_occurrence_joins(
        facts,
        occurrences,
        fact_columns,
    )
    if resolved_columns:
        aligned = [
            _select_for_table(matched_bytes, table_key),
            _select_for_table(fallback_join, table_key),
            _select_for_table(line_join, table_key),
        ]
    else:
        aligned = _align_frames_for_concat([matched_bytes, fallback_join, line_join])
    combined = pl.concat(aligned, how="vertical_relaxed")
    if resolved_columns:
        combined = combined.select(resolved_columns)
    combined = _dedupe_for_table(combined, table_key=table_key)
    reader = lazyframe_to_reader(combined)
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None:
        return reader
    return align_reader_to_contract(table_key, reader)


def _resolve_occurrence_joins(
    facts: pl.LazyFrame,
    occurrences: pl.LazyFrame,
    fact_columns: Sequence[str],
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    bytes_predicate = pl.col("start_byte").is_not_null() & pl.col("end_byte").is_not_null()
    facts_with_bytes = facts.filter(bytes_predicate)
    facts_without_bytes = facts.filter(~bytes_predicate)

    occ_bytes = occurrences.filter(
        pl.col("occ_start_byte").is_not_null() & pl.col("occ_end_byte").is_not_null()
    )
    bytes_join = join_validated(
        facts_with_bytes,
        occ_bytes,
        spec=_occurrence_byte_join_spec(),
    )
    fallback = bytes_join.filter(pl.col("scip_symbol").is_null()).select(fact_columns)
    line_join = _line_join_occurrences(facts_without_bytes, occurrences)
    fallback_join = _line_join_occurrences(fallback, occurrences)
    matched_bytes = bytes_join.filter(pl.col("scip_symbol").is_not_null())
    return matched_bytes, fallback_join, line_join


def _line_join_occurrences(left: pl.LazyFrame, occurrences: pl.LazyFrame) -> pl.LazyFrame:
    return join_validated(left, occurrences, spec=_occurrence_line_join_spec())


def _occurrence_byte_join_spec() -> JoinSpec:
    # Contract: occurrence spans are unique per byte span.
    return JoinSpec(
        left_on=["repo", "commit", "rel_path", "producer", "start_byte", "end_byte"],
        right_on=[
            "repo",
            "commit",
            "rel_path",
            "producer",
            "occ_start_byte",
            "occ_end_byte",
        ],
        how="left",
        validate="m:1",
    )


def _occurrence_line_join_spec() -> JoinSpec:
    # Contract: occurrence spans are unique per line/col span.
    return JoinSpec(
        left_on=[
            "repo",
            "commit",
            "rel_path",
            "producer",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
        ],
        right_on=[
            "repo",
            "commit",
            "rel_path",
            "producer",
            "occ_start_line",
            "occ_start_col",
            "occ_end_line",
            "occ_end_col",
        ],
        how="left",
        validate="m:1",
    )


def syntax_enrich__occurrence_resolution(
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
) -> pl.LazyFrame:
    """Return occurrence rows merged with SCIP roles and GOID metadata.

    Returns
    -------
    pl.LazyFrame
        LazyFrame containing merged occurrence metadata for resolution joins.
    """
    return _occurrence_resolution_frame(
        q__core__scip_occurrence_span_xref,
        q__core__scip_occurrence_syntax_xref,
    )


def syntax_enrich__defs_resolved__base(
    q__core__syntax_defs: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.RecordBatchReader:
    """Build core.syntax_defs_resolved from syntax defs and SCIP welds.

    Returns
    -------
    pa.RecordBatchReader
        Arrow reader for core.syntax_defs_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_defs)
    occurrences = tabular_to_lazyframe(syntax_enrich__occurrence_resolution)
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__refs_resolved__base(
    q__core__syntax_refs: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.RecordBatchReader:
    """Build core.syntax_refs_resolved from syntax refs and SCIP welds.

    Returns
    -------
    pa.RecordBatchReader
        Arrow reader for core.syntax_refs_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_refs)
    occurrences = tabular_to_lazyframe(syntax_enrich__occurrence_resolution)
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__calls_resolved__base(
    q__core__syntax_calls: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.RecordBatchReader:
    """Build core.syntax_calls_resolved from syntax calls and SCIP welds.

    Returns
    -------
    pa.RecordBatchReader
        Arrow reader for core.syntax_calls_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_calls)
    occurrences = tabular_to_lazyframe(syntax_enrich__occurrence_resolution)
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__imports_resolved__base(
    q__core__syntax_imports: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.RecordBatchReader:
    """Build core.syntax_imports_resolved from syntax imports and SCIP welds.

    Returns
    -------
    pa.RecordBatchReader
        Arrow reader for core.syntax_imports_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_imports)
    occurrences = tabular_to_lazyframe(syntax_enrich__occurrence_resolution)
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
    )


_MODULE = sys.modules[__name__]
_SYNTAX_ENRICH_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=SYNTAX_ENRICH_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__defs_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__defs_resolved",
            input_type=InferableTabularInput,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__refs_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__refs_resolved",
            input_type=InferableTabularInput,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__calls_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__calls_resolved",
            input_type=InferableTabularInput,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__imports_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__imports_resolved",
            input_type=InferableTabularInput,
        ),
    ),
    table_materializations_node="syntax_enrich__table_materializations",
    anchor_node_name="t__syntax_enrich",
)
attach_table_target_template(_MODULE, spec=_SYNTAX_ENRICH_TABLE_TARGET_SPEC)
syntax_enrich__defs_resolved = _MODULE.syntax_enrich__defs_resolved
syntax_enrich__refs_resolved = _MODULE.syntax_enrich__refs_resolved
syntax_enrich__calls_resolved = _MODULE.syntax_enrich__calls_resolved
syntax_enrich__imports_resolved = _MODULE.syntax_enrich__imports_resolved
syntax_enrich__table_materializations = _MODULE.syntax_enrich__table_materializations
t__syntax_enrich = _MODULE.t__syntax_enrich


__all__ = [
    "SYNTAX_CALLS_RESOLVED_TABLE_KEY",
    "SYNTAX_DEFS_RESOLVED_TABLE_KEY",
    "SYNTAX_ENRICH_TARGET_NAME",
    "SYNTAX_IMPORTS_RESOLVED_TABLE_KEY",
    "SYNTAX_REFS_RESOLVED_TABLE_KEY",
    "syntax_enrich__calls_resolved",
    "syntax_enrich__defs_resolved",
    "syntax_enrich__imports_resolved",
    "syntax_enrich__occurrence_resolution",
    "syntax_enrich__refs_resolved",
    "syntax_enrich__table_materializations",
    "t__syntax_enrich",
]
