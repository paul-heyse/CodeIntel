"""Resolved syntax fact tables welded with SCIP occurrences."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    RelationTableSaveSpec,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    align_table_to_contract,
    arrow_join_tables,
    build_join_options,
    dedupe_table_for_table,
)
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    invert_mask,
    is_null_mask,
    is_valid_mask,
)
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.table_ops import ensure_table_columns
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.columnar.schema_ops import concat_tables_unified

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


def _dedupe_for_table(table: pa.Table, *, table_key: str) -> pa.Table:
    return dedupe_table_for_table(table_key, table)


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
            arrays.append(pc.cast(column, pa.int64(), safe=False))
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
    span = tabular_to_arrow_table(q__core__scip_occurrence_span_xref).select(
        [
            "repo",
            "commit",
            "rel_path",
            "scip_symbol",
            "roles",
            "is_definition",
            "is_reference",
            "is_import",
            "is_write",
            "is_read",
            "goid_h128",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "start_byte",
            "end_byte",
        ]
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
    syntax = tabular_to_arrow_table(q__core__scip_occurrence_syntax_xref).select(
        [
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
        ]
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
    join_spec = ArrowJoinSpec(on=join_keys, how="left", validate="m:1")
    join_options = build_join_options(syntax, span)
    return arrow_join_tables(syntax, span, spec=join_spec, options=join_options)


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
    combined = _dedupe_for_table(combined, table_key=table_key)
    return align_table_to_contract(table_key, combined)


def _resolve_occurrence_joins(
    facts: pa.Table,
    occurrences: pa.Table,
    fact_columns: Sequence[str],
) -> tuple[pa.Table, pa.Table, pa.Table]:
    bytes_mask = _null_mask(facts, "start_byte", "end_byte")
    facts_with_bytes = _filter_table(facts, bytes_mask)
    facts_without_bytes = _filter_table(facts, invert_mask(bytes_mask))
    facts_with_bytes, extras_bytes = _detach_column(facts_with_bytes, "extras_json")
    facts_without_bytes, extras_no_bytes = _detach_column(facts_without_bytes, "extras_json")

    occ_bytes_mask = _null_mask(occurrences, "occ_start_byte", "occ_end_byte")
    occ_bytes = _filter_table(occurrences, occ_bytes_mask)
    byte_spec = _occurrence_byte_join_spec()
    facts_with_bytes = _cast_join_key_int64(facts_with_bytes, byte_spec.left_on)
    occ_bytes = _cast_join_key_int64(occ_bytes, byte_spec.right_on)
    join_options = build_join_options(facts_with_bytes, occ_bytes)
    bytes_join = arrow_join_tables(
        facts_with_bytes,
        occ_bytes,
        spec=byte_spec,
        options=join_options,
    )
    bytes_join = _attach_column(bytes_join, "extras_json", extras_bytes)
    fallback = _filter_table(bytes_join, is_null_mask(bytes_join["scip_symbol"]))
    fallback = fallback.select(fact_columns)
    line_join = _line_join_occurrences(facts_without_bytes, occurrences)
    line_join = _attach_column(line_join, "extras_json", extras_no_bytes)
    fallback_join = _line_join_occurrences(fallback, occurrences)
    matched_bytes = _filter_table(bytes_join, is_valid_mask(bytes_join["scip_symbol"]))
    return matched_bytes, fallback_join, line_join


def _null_mask(table: pa.Table, start_col: str, end_col: str) -> pa.BooleanArray:
    if start_col not in table.column_names or end_col not in table.column_names:
        return pa.array([False] * table.num_rows)
    return and_kleene(is_valid_mask(table[start_col]), is_valid_mask(table[end_col]))


def _filter_table(table: pa.Table, mask: pa.BooleanArray) -> pa.Table:
    if table.num_rows == 0:
        return table
    return table.filter(mask)


def _line_join_occurrences(left: pa.Table, occurrences: pa.Table) -> pa.Table:
    stripped_left, extras_json = _detach_column(left, "extras_json")
    line_spec = _occurrence_line_join_spec()
    stripped_left = _cast_join_key_int64(stripped_left, line_spec.left_on)
    occurrences = _cast_join_key_int64(occurrences, line_spec.right_on)
    join_options = build_join_options(stripped_left, occurrences)
    joined = arrow_join_tables(
        stripped_left,
        occurrences,
        spec=line_spec,
        options=join_options,
    )
    return _attach_column(joined, "extras_json", extras_json)


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


def _cast_join_key_int64(table: pa.Table, keys: Sequence[str] | None) -> pa.Table:
    if not keys:
        return table
    columns: list[pa.Array | pa.ChunkedArray] = []
    changed = False
    key_set = set(keys)
    for name in table.column_names:
        column = table[name]
        if name in key_set and pa.types.is_integer(column.type) and column.type != pa.int64():
            casted = _cast_to_int64(column)
            if casted is not column:
                column = casted
                changed = True
        columns.append(column)
    if not changed:
        return table
    return pa.Table.from_arrays(columns, names=list(table.column_names))


def _cast_to_int64(
    column: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    try:
        return pc.cast(column, pa.int64(), safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, ValueError):
        return column


def _occurrence_byte_join_spec() -> ArrowJoinSpec:
    # Contract: occurrence spans are unique per byte span.
    return ArrowJoinSpec(
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


def _occurrence_line_join_spec() -> ArrowJoinSpec:
    # Contract: occurrence spans are unique per line/col span.
    return ArrowJoinSpec(
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
    facts = tabular_to_arrow_table(q__core__syntax_defs)
    occurrences = tabular_to_arrow_table(syntax_enrich__occurrence_resolution)
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
    facts = tabular_to_arrow_table(q__core__syntax_refs)
    occurrences = tabular_to_arrow_table(syntax_enrich__occurrence_resolution)
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
    facts = tabular_to_arrow_table(q__core__syntax_calls)
    occurrences = tabular_to_arrow_table(syntax_enrich__occurrence_resolution)
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
    facts = tabular_to_arrow_table(q__core__syntax_imports)
    occurrences = tabular_to_arrow_table(syntax_enrich__occurrence_resolution)
    return _resolve_facts(
        facts,
        occurrences,
        table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
    )


_MODULE = sys.modules[__name__]
_SYNTAX_ENRICH_TABLE_TARGET_SPEC = build_multi_table_target_spec(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=SYNTAX_ENRICH_TARGET_NAME,
        tables=(
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext(
                    table_key=SYNTAX_DEFS_RESOLVED_TABLE_KEY,
                    base_node="syntax_enrich__defs_resolved__base",
                    node_name="syntax_enrich__defs_resolved",
                    input_type=InferableTabularInput,
                ),
                save_spec_factory=RelationTableSaveSpec,
                default_input_type=InferableTabularInput,
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext(
                    table_key=SYNTAX_REFS_RESOLVED_TABLE_KEY,
                    base_node="syntax_enrich__refs_resolved__base",
                    node_name="syntax_enrich__refs_resolved",
                    input_type=InferableTabularInput,
                ),
                save_spec_factory=RelationTableSaveSpec,
                default_input_type=InferableTabularInput,
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext(
                    table_key=SYNTAX_CALLS_RESOLVED_TABLE_KEY,
                    base_node="syntax_enrich__calls_resolved__base",
                    node_name="syntax_enrich__calls_resolved",
                    input_type=InferableTabularInput,
                ),
                save_spec_factory=RelationTableSaveSpec,
                default_input_type=InferableTabularInput,
            ),
            MultiTableTargetContext.build_table_spec(
                context=TableTargetTableContext(
                    table_key=SYNTAX_IMPORTS_RESOLVED_TABLE_KEY,
                    base_node="syntax_enrich__imports_resolved__base",
                    node_name="syntax_enrich__imports_resolved",
                    input_type=InferableTabularInput,
                ),
                save_spec_factory=RelationTableSaveSpec,
                default_input_type=InferableTabularInput,
            ),
        ),
        table_materializations_node="syntax_enrich__table_materializations",
        anchor_node_name="t__syntax_enrich",
        save_spec_factory=RelationTableSaveSpec,
        default_input_type=InferableTabularInput,
    )
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
