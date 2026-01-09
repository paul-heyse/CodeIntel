"""Resolved syntax fact tables welded with SCIP occurrences."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    RelationTableSaveSpec,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.ingestion_normalize import finalize_ingest_reader
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import normalize_table_for_join
from codeintel.build.tabular.compute_helpers import cast_array
from codeintel.build.tabular.conversion import table_to_reader, tabular_to_scoped_table
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.finalize_ops import (
    FinalizeDedupe,
    FinalizeResult,
    FinalizeSpec,
    finalize_join_keys,
    finalize_table,
    record_join_precheck_errors,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec, JoinType, Plan, materialize_plan
from codeintel.build.tabular.table_ops import ensure_table_columns
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.columnar.schema_ops import concat_tables_unified

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

LOG = logging.getLogger(__name__)

SYNTAX_ENRICH_TARGET_NAME = "syntax_enrich"
SYNTAX_DEFS_RESOLVED_TABLE_KEY = "core.syntax_defs_resolved"
SYNTAX_REFS_RESOLVED_TABLE_KEY = "core.syntax_refs_resolved"
SYNTAX_CALLS_RESOLVED_TABLE_KEY = "core.syntax_calls_resolved"
SYNTAX_IMPORTS_RESOLVED_TABLE_KEY = "core.syntax_imports_resolved"
SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY = "core.scip_occurrence_span_xref"
SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY = "core.scip_occurrence_syntax_xref"
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


def _select_for_table(table: pa.Table, table_key: str) -> pa.Table:
    columns = _ordered_columns(table_key)
    if not columns:
        return table
    return ensure_table_columns(table, columns).select(columns)


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


def _align_tables_for_concat(tables: list[pa.Table]) -> list[pa.Table]:
    schemas: list[list[str]] = [list(table.column_names) for table in tables]
    all_columns = _merge_column_names(schemas)
    aligned: list[pa.Table] = []
    for table, names in zip(tables, schemas, strict=True):
        missing = [name for name in all_columns if name not in names]
        resolved = table
        if missing:
            resolved = ensure_table_columns(resolved, [*names, *missing])
        aligned.append(resolved.select(all_columns))
    return aligned


_JOIN_STRING_KEYS = {
    "repo",
    "commit",
    "rel_path",
    "producer",
    "scip_symbol",
}
_JOIN_INT_KEYS = {
    "start_line",
    "start_col",
    "end_line",
    "end_col",
    "start_byte",
    "end_byte",
    "occ_start_line",
    "occ_start_col",
    "occ_end_line",
    "occ_end_col",
    "occ_start_byte",
    "occ_end_byte",
}


@dataclass(frozen=True, slots=True)
class _JoinSpec:
    left_keys: Sequence[str]
    right_keys: Sequence[str]
    left_table_key: str | None = None
    right_table_key: str | None = None


def _join_casts(keys: Sequence[str]) -> dict[str, str]:
    casts: dict[str, str] = {}
    for key in keys:
        if key in _JOIN_STRING_KEYS:
            casts[key] = "string"
        elif key in _JOIN_INT_KEYS:
            casts[key] = "int64"
    return casts


def _project_with_cast(
    table: pa.Table,
    *,
    casts: Mapping[str, str],
) -> dict[str, Expression]:
    exprs: dict[str, Expression] = {}
    for name in table.column_names:
        if name in casts:
            exprs[name] = E.cast(E.field(name), casts[name])
        else:
            exprs[name] = E.field(name)
    return exprs


def _precheck_join_table(
    table: pa.Table,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> pa.Table:
    if table.num_rows == 0 or not join_keys:
        return table
    if table_key is None:
        result = finalize_join_keys(
            table,
            required_non_null=join_keys,
            key_fields=join_keys,
        )
    else:
        result = finalize_table(
            table,
            spec=FinalizeSpec(
                table_key=table_key,
                mode="tolerant",
                required_non_null=join_keys,
                key_fields=join_keys,
                dedupe=FinalizeDedupe(enabled=False),
                target_name=SYNTAX_ENRICH_TARGET_NAME,
            ),
        )
    record_join_precheck_errors(
        result,
        table_key=table_key,
        target_name=SYNTAX_ENRICH_TARGET_NAME,
        join_keys=join_keys,
    )
    _log_join_precheck_errors(result, table_key=table_key, join_keys=join_keys)
    return result.good


def _log_join_precheck_errors(
    result: FinalizeResult,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> None:
    if result.errors.num_rows == 0:
        return
    table_label = table_key or "derived"
    LOG.warning(
        "Join key precheck dropped %d rows table=%s keys=%s",
        result.errors.num_rows,
        table_label,
        ",".join(join_keys),
    )


def _hash_join_tables(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: _JoinSpec,
    how: JoinType = "left outer",
) -> pa.Table:
    left_checked = _precheck_join_table(
        left,
        table_key=spec.left_table_key,
        join_keys=spec.left_keys,
    )
    right_checked = _precheck_join_table(
        right,
        table_key=spec.right_table_key,
        join_keys=spec.right_keys,
    )
    left_checked = normalize_table_for_join(left_checked)
    right_checked = normalize_table_for_join(right_checked)
    left_exprs = _project_with_cast(left_checked, casts=_join_casts(spec.left_keys))
    right_exprs = _project_with_cast(right_checked, casts=_join_casts(spec.right_keys))
    left_plan = Plan.table(left_checked).project(left_exprs)
    right_plan = Plan.table(right_checked).project(right_exprs)
    right_output = [name for name in right_exprs if name not in left_exprs]
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=list(spec.left_keys),
            right_keys=list(spec.right_keys),
            how=how,
            left_output=list(left_exprs.keys()),
            right_output=right_output,
        ),
    )
    joined = joined.order_by(sort_keys=[(key, "ascending") for key in spec.left_keys])
    return materialize_plan(joined, use_threads=True)


def _rename_columns(table: pa.Table, mapping: Mapping[str, str]) -> pa.Table:
    new_names = [mapping.get(name, name) for name in table.column_names]
    if new_names == list(table.column_names):
        return table
    return table.rename_columns(new_names)


def _coerce_occurrence_ints(table: pa.Table) -> pa.Table:
    arrays = []
    for name in table.column_names:
        column = table[name]
        if name in _OCCURRENCE_INT_COLUMNS:
            arrays.append(cast_array(column, pa.int64(), safe=False))
        else:
            arrays.append(column)
    return pa.Table.from_arrays(arrays, names=list(table.column_names))


def _drop_occurrence_bytes(table: pa.Table) -> pa.Table:
    drop_columns = [
        name for name in ("occ_start_byte", "occ_end_byte") if name in table.column_names
    ]
    if not drop_columns:
        return table
    return table.drop(drop_columns)


def _occurrence_resolution_table(
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
) -> pa.Table:
    span = tabular_to_scoped_table(
        q__core__scip_occurrence_span_xref,
        columns=[
            "repo",
            "commit",
            "rel_path",
            "scip_symbol",
            "roles",
            "syntax_kind",
            "is_definition",
            "is_reference",
            "is_import",
            "is_write",
            "is_read",
            "goid_h128",
            "documentation",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "start_byte",
            "end_byte",
        ],
        scope=None,
        require_scope_columns=False,
    )
    span = _rename_columns(
        span,
        {
            "roles": "scip_roles",
            "start_line": "occ_start_line",
            "start_col": "occ_start_col",
            "end_line": "occ_end_line",
            "end_col": "occ_end_col",
            "start_byte": "occ_start_byte",
            "end_byte": "occ_end_byte",
        },
    )
    span = _coerce_occurrence_ints(span)
    span = _drop_occurrence_bytes(span)
    syntax = tabular_to_scoped_table(
        q__core__scip_occurrence_syntax_xref,
        columns=[
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
        ],
        scope=None,
        require_scope_columns=False,
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
    return _hash_join_tables(
        syntax,
        span,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            left_table_key=SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY,
            right_table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
        ),
    )


def _resolve_facts(
    facts: pa.Table,
    occurrences: pa.Table,
    *,
    table_key: str,
) -> pa.Table:
    fact_columns = list(facts.column_names)
    if not fact_columns:
        return empty_table_for_table(table_key)
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
        aligned = _align_tables_for_concat([matched_bytes, fallback_join, line_join])
    tables = [table for table in aligned if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(table_key)
    combined = concat_tables_unified(tables)
    if resolved_columns:
        combined = combined.select(resolved_columns)
    reader = table_to_reader(combined, batch_size=None)
    return finalize_ingest_reader(
        table_key,
        reader,
        target_name=SYNTAX_ENRICH_TARGET_NAME,
    )


def _resolve_occurrence_joins(
    facts: pa.Table,
    occurrences: pa.Table,
    fact_columns: Sequence[str],
) -> tuple[pa.Table, pa.Table, pa.Table]:
    bytes_expr = _valid_pair_expr("start_byte", "end_byte")
    facts_with_bytes = _filter_table_expr(facts, bytes_expr)
    facts_without_bytes = _filter_table_expr(facts, ~bytes_expr)
    facts_with_bytes, extras_bytes = _detach_column(facts_with_bytes, "extras")
    facts_without_bytes, extras_no_bytes = _detach_column(facts_without_bytes, "extras")

    occ_bytes_expr = _valid_pair_expr("occ_start_byte", "occ_end_byte")
    occ_bytes = _filter_table_expr(occurrences, occ_bytes_expr)
    byte_left, byte_right = _occurrence_byte_join_keys()
    bytes_join = _hash_join_tables(
        facts_with_bytes,
        occ_bytes,
        spec=_JoinSpec(left_keys=byte_left, right_keys=byte_right),
    )
    bytes_join = _attach_column(bytes_join, "extras", extras_bytes)
    fallback = _filter_table_expr(bytes_join, E.is_null("scip_symbol"))
    fallback = fallback.select(fact_columns)
    line_join = _line_join_occurrences(facts_without_bytes, occurrences)
    line_join = _attach_column(line_join, "extras", extras_no_bytes)
    fallback_join = _line_join_occurrences(fallback, occurrences)
    matched_bytes = _filter_table_expr(bytes_join, E.is_valid("scip_symbol"))
    return matched_bytes, fallback_join, line_join


def _valid_pair_expr(start_col: str, end_col: str) -> Expression:
    return E.and_(E.is_valid(start_col), E.is_valid(end_col))


def _filter_table_expr(table: pa.Table, expr: Expression) -> pa.Table:
    if table.num_rows == 0:
        return table
    return materialize_plan(Plan.table(table).filter(expr), use_threads=True)


def _line_join_occurrences(left: pa.Table, occurrences: pa.Table) -> pa.Table:
    stripped_left, extras = _detach_column(left, "extras")
    line_left, line_right = _occurrence_line_join_keys()
    joined = _hash_join_tables(
        stripped_left,
        occurrences,
        spec=_JoinSpec(left_keys=line_left, right_keys=line_right),
    )
    return _attach_column(joined, "extras", extras)


def _detach_column(
    table: pa.Table,
    column_name: str,
) -> tuple[pa.Table, pa.Array | pa.ChunkedArray | None]:
    if column_name not in table.column_names:
        return table, None
    return table.drop_columns([column_name]), table[column_name]


def _attach_column(
    table: pa.Table,
    column_name: str,
    column: pa.Array | pa.ChunkedArray | None,
) -> pa.Table:
    if column is None:
        return table
    return table.append_column(column_name, column)


def _occurrence_byte_join_keys() -> tuple[list[str], list[str]]:
    # Contract: occurrence spans are unique per byte span.
    return (
        ["repo", "commit", "rel_path", "producer", "start_byte", "end_byte"],
        [
            "repo",
            "commit",
            "rel_path",
            "producer",
            "occ_start_byte",
            "occ_end_byte",
        ],
    )


def _occurrence_line_join_keys() -> tuple[list[str], list[str]]:
    # Contract: occurrence spans are unique per line/col span.
    return (
        [
            "repo",
            "commit",
            "rel_path",
            "producer",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
        ],
        [
            "repo",
            "commit",
            "rel_path",
            "producer",
            "occ_start_line",
            "occ_start_col",
            "occ_end_line",
            "occ_end_col",
        ],
    )


def syntax_enrich__occurrence_resolution(
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
) -> pa.Table:
    """Return occurrence rows merged with SCIP roles and GOID metadata.

    Returns
    -------
    pa.Table
        Arrow table containing merged occurrence metadata for resolution joins.
    """
    return _occurrence_resolution_table(
        q__core__scip_occurrence_span_xref,
        q__core__scip_occurrence_syntax_xref,
    )


def syntax_enrich__defs_resolved__base(
    q__core__syntax_defs: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.Table:
    """Build core.syntax_defs_resolved from syntax defs and SCIP welds.

    Returns
    -------
    pa.Table
        Arrow reader for core.syntax_defs_resolved.
    """
    facts = tabular_to_scoped_table(
        q__core__syntax_defs,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    occurrences = tabular_to_scoped_table(
        syntax_enrich__occurrence_resolution,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__refs_resolved__base(
    q__core__syntax_refs: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.Table:
    """Build core.syntax_refs_resolved from syntax refs and SCIP welds.

    Returns
    -------
    pa.Table
        Arrow reader for core.syntax_refs_resolved.
    """
    facts = tabular_to_scoped_table(
        q__core__syntax_refs,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    occurrences = tabular_to_scoped_table(
        syntax_enrich__occurrence_resolution,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__calls_resolved__base(
    q__core__syntax_calls: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.Table:
    """Build core.syntax_calls_resolved from syntax calls and SCIP welds.

    Returns
    -------
    pa.Table
        Arrow reader for core.syntax_calls_resolved.
    """
    facts = tabular_to_scoped_table(
        q__core__syntax_calls,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    occurrences = tabular_to_scoped_table(
        syntax_enrich__occurrence_resolution,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
    )


def syntax_enrich__imports_resolved__base(
    q__core__syntax_imports: InferableTabularInput,
    syntax_enrich__occurrence_resolution: InferableTabularInput,
) -> pa.Table:
    """Build core.syntax_imports_resolved from syntax imports and SCIP welds.

    Returns
    -------
    pa.Table
        Arrow reader for core.syntax_imports_resolved.
    """
    facts = tabular_to_scoped_table(
        q__core__syntax_imports,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    occurrences = tabular_to_scoped_table(
        syntax_enrich__occurrence_resolution,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
    )


_MODULE = sys.modules[__name__]
_SYNTAX_ENRICH_TABLE_CONTEXTS = (
    TableTargetTableContext(
        table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY,
        base_node="syntax_enrich__defs_resolved__base",
        node_name="syntax_enrich__defs_resolved",
    ),
    TableTargetTableContext(
        table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
        base_node="syntax_enrich__refs_resolved__base",
        node_name="syntax_enrich__refs_resolved",
    ),
    TableTargetTableContext(
        table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
        base_node="syntax_enrich__calls_resolved__base",
        node_name="syntax_enrich__calls_resolved",
    ),
    TableTargetTableContext(
        table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
        base_node="syntax_enrich__imports_resolved__base",
        node_name="syntax_enrich__imports_resolved",
    ),
)
_SYNTAX_ENRICH_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=SYNTAX_ENRICH_TARGET_NAME,
        tables=(),
        table_materializations_node="syntax_enrich__table_materializations",
        anchor_node_name="t__syntax_enrich",
        save_spec_factory=RelationTableSaveSpec,
        default_input_type=InferableTabularInput,
    ),
    table_contexts=_SYNTAX_ENRICH_TABLE_CONTEXTS,
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
