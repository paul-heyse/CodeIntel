"""Resolved syntax fact tables welded with SCIP occurrences."""

from __future__ import annotations

import sys

import polars as pl

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
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import (
    dedupe_frame_for_table,
    empty_lazyframe_for_table,
)
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SYNTAX_ENRICH_TARGET_NAME = "syntax_enrich"
SYNTAX_DEFS_RESOLVED_TABLE_KEY = "core.syntax_defs_resolved"
SYNTAX_REFS_RESOLVED_TABLE_KEY = "core.syntax_refs_resolved"
SYNTAX_CALLS_RESOLVED_TABLE_KEY = "core.syntax_calls_resolved"
SYNTAX_IMPORTS_RESOLVED_TABLE_KEY = "core.syntax_imports_resolved"


def _ordered_columns(table_key: str) -> list[str]:
    schema = get_schema_service().require_table_schema(table_key)
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


def _occurrence_resolution_frame(
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
) -> pl.LazyFrame:
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
    return syntax.join(span, on=join_keys, how="left")


def _resolve_facts(
    facts: pl.LazyFrame,
    occurrences: pl.LazyFrame,
    *,
    table_key: str,
) -> pl.LazyFrame:
    fact_columns = list(facts.collect_schema().names())
    if not fact_columns:
        return empty_lazyframe_for_table(table_key)
    resolved_columns = _ordered_columns(table_key)

    bytes_predicate = pl.col("start_byte").is_not_null() & pl.col("end_byte").is_not_null()
    facts_with_bytes = facts.filter(bytes_predicate)
    facts_without_bytes = facts.filter(~bytes_predicate)

    occ_bytes = occurrences.filter(
        pl.col("occ_start_byte").is_not_null() & pl.col("occ_end_byte").is_not_null()
    )
    bytes_join = facts_with_bytes.join(
        occ_bytes,
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
    )

    fallback = bytes_join.filter(pl.col("scip_symbol").is_null()).select(fact_columns)
    line_join = facts_without_bytes.join(
        occurrences,
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
    )
    fallback_join = fallback.join(
        occurrences,
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
    )

    matched_bytes = bytes_join.filter(pl.col("scip_symbol").is_not_null())
    aligned = [
        _select_for_table(matched_bytes, table_key),
        _select_for_table(fallback_join, table_key),
        _select_for_table(line_join, table_key),
    ]
    combined = pl.concat(aligned, how="vertical_relaxed")
    if resolved_columns:
        combined = combined.select(resolved_columns)
    return dedupe_frame_for_table(combined, table_key=table_key)


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
    syntax_enrich__occurrence_resolution: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build core.syntax_defs_resolved from syntax defs and SCIP welds.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for core.syntax_defs_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_defs)
    return _resolve_facts(
        facts,
        syntax_enrich__occurrence_resolution,
        table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__refs_resolved__base(
    q__core__syntax_refs: InferableTabularInput,
    syntax_enrich__occurrence_resolution: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build core.syntax_refs_resolved from syntax refs and SCIP welds.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for core.syntax_refs_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_refs)
    return _resolve_facts(
        facts,
        syntax_enrich__occurrence_resolution,
        table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__calls_resolved__base(
    q__core__syntax_calls: InferableTabularInput,
    syntax_enrich__occurrence_resolution: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build core.syntax_calls_resolved from syntax calls and SCIP welds.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for core.syntax_calls_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_calls)
    return _resolve_facts(
        facts,
        syntax_enrich__occurrence_resolution,
        table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__imports_resolved__base(
    q__core__syntax_imports: InferableTabularInput,
    syntax_enrich__occurrence_resolution: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build core.syntax_imports_resolved from syntax imports and SCIP welds.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for core.syntax_imports_resolved.
    """
    facts = tabular_to_lazyframe(q__core__syntax_imports)
    return _resolve_facts(
        facts,
        syntax_enrich__occurrence_resolution,
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
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__refs_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__refs_resolved",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__calls_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__calls_resolved",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
            base_node="syntax_enrich__imports_resolved__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY),
            node_name="syntax_enrich__imports_resolved",
            input_type=pl.LazyFrame,
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
