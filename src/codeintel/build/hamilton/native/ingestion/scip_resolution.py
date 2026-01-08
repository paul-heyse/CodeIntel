"""SCIP resolution tables for deterministic symbol/GOID stitching."""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa
from google.protobuf.struct_pb2 import NullValue, Struct

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
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    align_table_to_contract,
    arrow_join_tables,
    build_join_options,
    dedupe_table_for_table,
    emit_alignment_report,
    iter_rows,
)
from codeintel.build.tabular.compute_columns import constant_array
from codeintel.build.tabular.compute_helpers import (
    array_from_compute,
    cast_array,
    safe_filter,
)
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    bit_wise_and,
    equal_mask,
    invert_mask,
    is_valid_mask,
    not_equal_mask,
)
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.columnar.schema_ops import concat_tables_unified
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    DagCatalog,
    TargetRunRecord,
    InferableTabularInput,
)
LOG = logging.getLogger(__name__)

SCIP_RESOLUTION_TARGET_NAME = "scip_resolution"
SCIP_SYMBOL_GOID_XREF_TABLE_KEY = "core.scip_symbol_goid_xref"
SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY = "core.scip_occurrence_span_xref"
SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY = "core.scip_occurrence_syntax_xref"

_ROLE_DEFINITION = 0x1
_ROLE_IMPORT = 0x2
_ROLE_WRITE = 0x4
_ROLE_READ = 0x8

_MATCH_KIND_BYTE_SPAN = "byte_span"
_MATCH_KIND_LINE_COL = "line_col"
_MATCH_KIND_LINE_SPAN = "line_span"
_MATCH_KIND_LINE_START = "line_start"
_MATCH_CONFIDENCE_BYTE_SPAN = 1.0
_MATCH_CONFIDENCE_LINE_COL = 0.8
_MATCH_CONFIDENCE_LINE_SPAN = 1.0
_MATCH_CONFIDENCE_LINE_START = 0.6


@dataclass(frozen=True)
class ScipResolutionFrames:
    """Derived frames for SCIP resolution outputs."""

    symbol_goid_xref: pa.Table
    occurrence_span_xref: pa.Table


@dataclass(slots=True)
class _SyntaxNodeIndex:
    resolver: SpanResolver[str]
    line_exact: dict[tuple[int, int, int, int], list[str]]


def _rename_columns(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    new_names = [mapping.get(name, name) for name in table.column_names]
    if new_names == list(table.column_names):
        return table
    return table.rename_columns(new_names)


def _cast_int32(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    arrays = []
    for name in table.column_names:
        column = table[name]
        if name in columns:
            arrays.append(cast_array(column, pa.int32(), safe=False))
        else:
            arrays.append(column)
    return pa.Table.from_arrays(arrays, names=list(table.column_names))


def _drop_columns_if_present(table: pa.Table, names: Sequence[str]) -> pa.Table:
    existing = [name for name in names if name in table.column_names]
    if not existing:
        return table
    return table.drop_columns(existing)


def _append_null_column(table: pa.Table, name: str, data_type: pa.DataType) -> pa.Table:
    if name in table.column_names:
        return table
    if table.num_rows == 0:
        return table.append_column(name, pa.array([], type=data_type))
    return table.append_column(name, pa.nulls(table.num_rows, type=data_type))


def _valid_mask_for_columns(
    table: pa.Table, columns: Sequence[str]
) -> pa.Array | pa.ChunkedArray | None:
    mask: pa.Array | pa.ChunkedArray | None = None
    for name in columns:
        column_mask = is_valid_mask(table[name])
        mask = column_mask if mask is None else and_kleene(mask, column_mask)
    return mask


def _split_by_valid_columns(table: pa.Table, columns: Sequence[str]) -> tuple[pa.Table, pa.Table]:
    if table.num_rows == 0:
        return table, table
    mask = _valid_mask_for_columns(table, columns)
    if mask is None:
        return table, table.slice(0, 0)
    valid = safe_filter(table, mask)
    invalid = safe_filter(table, invert_mask(mask))
    return valid, invalid


def _goid_type_for_table(table: pa.Table) -> pa.DataType:
    if "goid_h128" in table.column_names:
        return table.schema.field("goid_h128").type
    return pa.decimal128(38, 0)


def _replace_column(
    table: pa.Table,
    name: str,
    values: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    return table.set_column(index, name, values)


def _if_else(
    condition: pa.Array | pa.ChunkedArray,
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    result = array_from_compute("if_else", [condition, left, right])
    if result is None:
        msg = "Arrow compute if_else did not return an array."
        raise TypeError(msg)
    return result


def _empty_reader_for_output_table(table_key: str) -> pa.Table:
    try:
        return empty_table_for_table(table_key)
    except KeyError:
        table_schema = OUTPUT_TABLE_SCHEMAS.get(table_key)
        if table_schema is None:
            raise
        arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
        return pa.Table.from_batches(arrow_schema, [])


def _empty_table_for_output_table(table_key: str) -> pa.Table:
    table_schema = OUTPUT_TABLE_SCHEMAS.get(table_key)
    if table_schema is None:
        msg = f"Missing output schema for {table_key}"
        raise KeyError(msg)
    arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    return pa.Table.from_batches([], schema=arrow_schema)


def _apply_enclosing_ranges(table: pa.Table) -> pa.Table:
    required = {
        "enclosing_start_line",
        "enclosing_start_col",
        "enclosing_end_line",
        "enclosing_end_col",
    }
    if table.num_rows == 0 or not required.issubset(set(table.column_names)):
        return table
    enclosing_mask = and_kleene(
        and_kleene(
            is_valid_mask(table["enclosing_start_line"]),
            is_valid_mask(table["enclosing_start_col"]),
        ),
        and_kleene(
            is_valid_mask(table["enclosing_end_line"]),
            is_valid_mask(table["enclosing_end_col"]),
        ),
    )
    table = _replace_column(
        table,
        "start_line",
        _if_else(enclosing_mask, table["enclosing_start_line"], table["start_line"]),
    )
    table = _replace_column(
        table,
        "start_col",
        _if_else(enclosing_mask, table["enclosing_start_col"], table["start_col"]),
    )
    table = _replace_column(
        table,
        "end_line",
        _if_else(enclosing_mask, table["enclosing_end_line"], table["end_line"]),
    )
    return _replace_column(
        table,
        "end_col",
        _if_else(enclosing_mask, table["enclosing_end_col"], table["end_col"]),
    )


def _apply_occurrence_documentation(table: pa.Table) -> pa.Table:
    if table.num_rows == 0:
        return table
    if "override_documentation" not in table.column_names:
        return table
    override_doc = table["override_documentation"]
    if "documentation" not in table.column_names:
        return table.append_column("documentation", override_doc)
    merged_doc = _if_else(
        is_valid_mask(override_doc),
        override_doc,
        table["documentation"],
    )
    return _replace_column(table, "documentation", merged_doc)


def _symbol_info_table(symbol_info: InferableTabularInput) -> pa.Table:
    table = tabular_to_scoped_table(
        symbol_info,
        columns=[
            "repo",
            "commit",
            "symbol",
            "documentation",
            "enclosing_symbol",
        ],
        scope=None,
        require_scope_columns=False,
    )
    return _rename_columns(table, {"symbol": "scip_symbol"})


def _goids_table(goids: InferableTabularInput) -> pa.Table:
    table = tabular_to_scoped_table(
        goids,
        columns=[
            "goid_h128",
            "rel_path",
            "start_line",
            "end_line",
        ],
        scope=None,
        require_scope_columns=False,
    )
    table = _cast_int32(table, ["start_line", "end_line"])
    if table.num_rows == 0:
        return table
    mask = and_kleene(
        is_valid_mask(table["start_line"]),
        is_valid_mask(table["end_line"]),
    )
    return safe_filter(table, mask)


def _definition_anchors_table(defs_resolved: InferableTabularInput) -> pa.Table:
    table = tabular_to_scoped_table(
        defs_resolved,
        columns=[
            "repo",
            "commit",
            "rel_path",
            "goid_h128",
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
    table = _cast_int32(table, ["start_line", "start_col", "end_line", "end_col"])
    if table.num_rows == 0:
        return table
    return safe_filter(table, is_valid_mask(table["goid_h128"]))


def _occurrences_table(occurrences: InferableTabularInput) -> pa.Table:
    table = tabular_to_scoped_table(
        occurrences,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    table = _rename_columns(table, {"symbol": "scip_symbol"})
    return _cast_int32(
        table,
        [
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "enclosing_start_line",
            "enclosing_start_col",
            "enclosing_end_line",
            "enclosing_end_col",
        ],
    )


def _goids_by_start_line(goids: pa.Table) -> pa.Table:
    required = {"rel_path", "start_line", "goid_h128"}
    if goids.num_rows == 0 or not required.issubset(set(goids.column_names)):
        return goids
    grouped = goids.group_by(["rel_path", "start_line"]).aggregate([("goid_h128", "min")])
    return grouped.rename_columns(["rel_path", "start_line", "goid_h128"])


def _attach_match_metadata(
    table: pa.Table,
    *,
    kind: str | None,
    confidence: float | None,
) -> pa.Table:
    if table.num_rows == 0:
        return table
    match_kind = constant_array(kind, table.num_rows)
    match_confidence = constant_array(confidence, table.num_rows)
    table = table.append_column("match_kind", match_kind)
    return table.append_column("match_confidence", match_confidence)


def _definition_occurrences(occurrences: pa.Table) -> pa.Table:
    roles = occurrences["roles"] if "roles" in occurrences.column_names else None
    if roles is None:
        return occurrences.slice(0, 0)
    def_mask = not_equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_DEFINITION, type=roles.type)),
        pa.scalar(0, type=roles.type),
    )
    return safe_filter(occurrences, def_mask)


def _anchor_goid_matches(
    definitions: pa.Table,
    anchors: pa.Table,
    *,
    join_keys: Sequence[str],
    match_kind: str,
    confidence: float,
) -> tuple[pa.Table, pa.Table]:
    matched = definitions.slice(0, 0)
    missing = definitions
    required_right = set(join_keys) | {"goid_h128"}
    can_join = True
    if (
        definitions.num_rows == 0
        or anchors.num_rows == 0
        or not set(join_keys).issubset(definitions.column_names)
        or not required_right.issubset(anchors.column_names)
    ):
        can_join = False
    if can_join:
        left = _drop_columns_if_present(
            definitions,
            ["goid_h128", "match_kind", "match_confidence"],
        )
        left_valid, left_invalid = _split_by_valid_columns(left, join_keys)
        right_mask = _valid_mask_for_columns(anchors, list(required_right))
        if left_valid.num_rows == 0 or right_mask is None:
            can_join = False
        else:
            anchors_valid = safe_filter(anchors, right_mask)
            if anchors_valid.num_rows == 0:
                can_join = False
            else:
                join_spec = ArrowJoinSpec(on=list(join_keys), how="left", validate="m:1")
                joined = arrow_join_tables(
                    left_valid,
                    anchors_valid,
                    spec=join_spec,
                    options=build_join_options(left_valid, anchors_valid),
                )
                matched_mask = is_valid_mask(joined["goid_h128"])
                matched = safe_filter(joined, matched_mask)
                matched = _attach_match_metadata(
                    matched,
                    kind=match_kind,
                    confidence=confidence,
                )
                missing = safe_filter(joined, invert_mask(matched_mask))
                if left_invalid.num_rows != 0:
                    goid_type = _goid_type_for_table(anchors_valid)
                    left_invalid = _append_null_column(left_invalid, "goid_h128", goid_type)
                    missing = concat_tables_unified([missing, left_invalid])
    return matched, missing


def _byte_span_goid_matches(
    definitions: pa.Table,
    anchors: pa.Table,
) -> tuple[pa.Table, pa.Table]:
    anchor_cols = ["rel_path", "start_byte", "end_byte", "goid_h128"]
    if not set(anchor_cols).issubset(anchors.column_names):
        return definitions.slice(0, 0), definitions
    return _anchor_goid_matches(
        definitions,
        anchors.select(anchor_cols),
        join_keys=["rel_path", "start_byte", "end_byte"],
        match_kind=_MATCH_KIND_BYTE_SPAN,
        confidence=_MATCH_CONFIDENCE_BYTE_SPAN,
    )


def _line_col_goid_matches(
    definitions: pa.Table,
    anchors: pa.Table,
) -> tuple[pa.Table, pa.Table]:
    anchor_cols = [
        "rel_path",
        "start_line",
        "start_col",
        "end_line",
        "end_col",
        "goid_h128",
    ]
    if not set(anchor_cols).issubset(anchors.column_names):
        return definitions.slice(0, 0), definitions
    return _anchor_goid_matches(
        definitions,
        anchors.select(anchor_cols),
        join_keys=["rel_path", "start_line", "start_col", "end_line", "end_col"],
        match_kind=_MATCH_KIND_LINE_COL,
        confidence=_MATCH_CONFIDENCE_LINE_COL,
    )


def _strict_goid_matches(
    definitions: pa.Table,
    goids: pa.Table,
) -> tuple[pa.Table, pa.Table]:
    required = {"rel_path", "start_line", "end_line"}
    if definitions.num_rows == 0 or goids.num_rows == 0:
        return definitions.slice(0, 0), definitions
    if not required.issubset(set(definitions.column_names)):
        return definitions.slice(0, 0), definitions
    if not required.issubset(set(goids.column_names)):
        return definitions.slice(0, 0), definitions
    left = _drop_columns_if_present(
        definitions,
        ["goid_h128", "match_kind", "match_confidence"],
    )
    join_spec = ArrowJoinSpec(on=["rel_path", "start_line", "end_line"], how="left", validate="m:1")
    strict_join = arrow_join_tables(
        left,
        goids,
        spec=join_spec,
        options=build_join_options(left, goids),
    )
    strict_mask = is_valid_mask(strict_join["goid_h128"])
    strict_matched = safe_filter(strict_join, strict_mask)
    strict_matched = _attach_match_metadata(
        strict_matched,
        kind=_MATCH_KIND_LINE_SPAN,
        confidence=_MATCH_CONFIDENCE_LINE_SPAN,
    )
    strict_missing = safe_filter(strict_join, invert_mask(strict_mask))
    return strict_matched, strict_missing


def _fallback_goid_matches(definitions: pa.Table, goids: pa.Table) -> list[pa.Table]:
    if definitions.num_rows == 0:
        return []
    fallback_left = _drop_columns_if_present(
        definitions,
        ["goid_h128", "match_kind", "match_confidence"],
    )
    goids_by_line = _goids_by_start_line(goids)
    fallback_spec = ArrowJoinSpec(
        on=["rel_path", "start_line"],
        how="left",
        validate="m:1",
    )
    fallback_join = arrow_join_tables(
        fallback_left,
        goids_by_line,
        spec=fallback_spec,
        options=build_join_options(fallback_left, goids_by_line),
    )
    fallback_mask = is_valid_mask(fallback_join["goid_h128"])
    fallback_matched = safe_filter(fallback_join, fallback_mask)
    fallback_unmatched = safe_filter(fallback_join, invert_mask(fallback_mask))
    fallback_matched = _attach_match_metadata(
        fallback_matched,
        kind=_MATCH_KIND_LINE_START,
        confidence=_MATCH_CONFIDENCE_LINE_START,
    )
    fallback_unmatched = _attach_match_metadata(
        fallback_unmatched,
        kind=None,
        confidence=None,
    )
    return [fallback_matched, fallback_unmatched]


def _log_symbol_goid_coverage(table: pa.Table) -> None:
    if table.num_rows == 0 or "match_kind" not in table.column_names:
        return
    counts: dict[str, int] = {}
    for value in table.column("match_kind").to_pylist():
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    summary = " ".join(f"{key}={count}" for key, count in sorted(counts.items()))
    LOG.info(
        "scip_symbol_goid_xref match coverage total=%d %s",
        table.num_rows,
        summary,
    )


def _symbol_goid_xref_table(
    *,
    occurrences: pa.Table,
    goids: pa.Table,
    anchors: pa.Table,
    created_at: datetime,
) -> pa.Table:
    if occurrences.num_rows == 0:
        return _empty_table_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    definitions = _definition_occurrences(occurrences)
    if definitions.num_rows == 0:
        return _empty_table_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    if anchors.num_rows == 0 and goids.num_rows == 0:
        return _empty_table_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    matched_tables: list[pa.Table] = []
    remaining = definitions
    if anchors.num_rows != 0:
        byte_matched, remaining = _byte_span_goid_matches(remaining, anchors)
        if byte_matched.num_rows != 0:
            matched_tables.append(byte_matched)
        line_col_matched, remaining = _line_col_goid_matches(remaining, anchors)
        if line_col_matched.num_rows != 0:
            matched_tables.append(line_col_matched)
    fallback_tables: list[pa.Table] = []
    if goids.num_rows != 0:
        strict_matched, strict_missing = _strict_goid_matches(remaining, goids)
        if strict_matched.num_rows != 0:
            matched_tables.append(strict_matched)
        fallback_tables = _fallback_goid_matches(strict_missing, goids)
    joined_tables = [*matched_tables, *fallback_tables]
    if not joined_tables:
        return _empty_table_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    joined = concat_tables_unified(joined_tables)
    joined = joined.select(
        [
            "repo",
            "commit",
            "scip_symbol",
            "goid_h128",
            "rel_path",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "position_encoding",
            "text_document_encoding",
            "match_kind",
            "match_confidence",
        ]
    )
    joined = _rename_columns(
        joined,
        {
            "rel_path": "def_rel_path",
            "start_line": "def_start_line",
            "start_col": "def_start_col",
            "end_line": "def_end_line",
            "end_col": "def_end_col",
        },
    )
    _log_symbol_goid_coverage(joined)
    created = constant_array(created_at, joined.num_rows)
    return joined.append_column("created_at", created)


def _occurrence_span_xref_table(
    *,
    occurrences: pa.Table,
    symbol_info: pa.Table,
    symbol_goid_xref: pa.Table,
    created_at: datetime,
) -> pa.Table:
    goid_lookup_source = dedupe_table_for_table(
        SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
        symbol_goid_xref,
    )
    goid_lookup = goid_lookup_source.select(
        [
            "repo",
            "commit",
            "scip_symbol",
            "goid_h128",
        ]
    )
    join_spec = ArrowJoinSpec(
        on=["repo", "commit", "scip_symbol"],
        how="left",
        validate="m:1",
    )
    join_options = build_join_options(occurrences, symbol_info)
    base = arrow_join_tables(
        occurrences,
        symbol_info,
        spec=join_spec,
        options=join_options,
    )
    join_options = build_join_options(base, goid_lookup)
    base = arrow_join_tables(
        base,
        goid_lookup,
        spec=join_spec,
        options=join_options,
    )
    base = _apply_occurrence_documentation(base)
    base = _apply_enclosing_ranges(base)
    roles = base["roles"] if "roles" in base.column_names else None
    if roles is None:
        return _empty_table_for_output_table(SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY)
    role_scalar = pa.scalar(0, type=roles.type)
    is_definition = not_equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_DEFINITION, type=roles.type)),
        role_scalar,
    )
    is_import = not_equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_IMPORT, type=roles.type)),
        role_scalar,
    )
    is_write = not_equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_WRITE, type=roles.type)),
        role_scalar,
    )
    is_read = not_equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_READ, type=roles.type)),
        role_scalar,
    )
    is_reference = equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_DEFINITION, type=roles.type)),
        role_scalar,
    )
    base = base.append_column("is_definition", is_definition)
    base = base.append_column("is_reference", is_reference)
    base = base.append_column("is_import", is_import)
    base = base.append_column("is_write", is_write)
    base = base.append_column("is_read", is_read)
    if "created_at" in base.column_names:
        base = base.drop_columns(["created_at"])
    created = constant_array(created_at, base.num_rows)
    base = base.append_column("created_at", created)
    return base.select(
        [
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
            "enclosing_symbol",
            "documentation",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "position_encoding",
            "text_document_encoding",
            "start_byte",
            "end_byte",
            "goid_h128",
            "created_at",
        ]
    )


def _stable_occurrence_id(row: dict[str, object]) -> str:
    msg = Struct()
    fields = (
        "rel_path",
        "scip_symbol",
        "occ_start_line",
        "occ_start_col",
        "occ_end_line",
        "occ_end_col",
        "occ_start_byte",
        "occ_end_byte",
    )
    for name in fields:
        value = row.get(name)
        if value is None:
            msg.fields[name].null_value = NullValue.NULL_VALUE
            continue
        msg.fields[name].string_value = str(value)
    payload = msg.SerializeToString(deterministic=True)
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _build_syntax_node_indexes(
    nodes_table: pa.Table,
) -> dict[tuple[str, str], _SyntaxNodeIndex]:
    indexes: dict[tuple[str, str], _SyntaxNodeIndex] = {}
    for row in iter_rows(nodes_table):
        rel_path = row.get("rel_path")
        producer = row.get("producer")
        node_id = row.get("node_id")
        if not isinstance(rel_path, str) or not isinstance(producer, str):
            continue
        if not isinstance(node_id, str):
            continue
        key = (rel_path, producer)
        index = indexes.get(key)
        if index is None:
            index = _SyntaxNodeIndex(
                resolver=SpanResolver.for_bytes(path_normalizer=lambda value: value),
                line_exact={},
            )
            indexes[key] = index
        start_line = row.get("start_line")
        start_col = row.get("start_col")
        end_line = row.get("end_line")
        end_col = row.get("end_col")
        if (
            isinstance(start_line, int)
            and isinstance(start_col, int)
            and isinstance(end_line, int)
            and isinstance(end_col, int)
        ):
            line_key = (start_line, start_col, end_line, end_col)
            index.line_exact.setdefault(line_key, []).append(node_id)
        start_byte = row.get("start_byte")
        end_byte = row.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int):
            index.resolver.add_span(rel_path, start_byte, end_byte, node_id)
    return indexes


def _match_occurrence_to_node(
    index: _SyntaxNodeIndex,
    rel_path: str,
    occ_row: dict[str, object],
) -> tuple[str | None, str, int]:
    start_byte = occ_row.get("occ_start_byte")
    end_byte = occ_row.get("occ_end_byte")
    if (
        isinstance(start_byte, int)
        and isinstance(end_byte, int)
        and start_byte >= 0
        and end_byte >= 0
    ):
        match = index.resolver.resolve(
            rel_path,
            start_byte,
            end_byte,
            allow_adjacent_point=True,
        )
        if match.match_kind != "NONE":
            return match.payload, match.match_kind, match.candidate_count

    start_line = occ_row.get("occ_start_line")
    start_col = occ_row.get("occ_start_col")
    end_line = occ_row.get("occ_end_line")
    end_col = occ_row.get("occ_end_col")
    if (
        isinstance(start_line, int)
        and isinstance(start_col, int)
        and isinstance(end_line, int)
        and isinstance(end_col, int)
    ):
        line_key = (start_line, start_col, end_line, end_col)
        exact = index.line_exact.get(line_key)
        if exact:
            return min(exact), "EXACT", len(exact)
    return None, "NONE", 0


def _occurrence_syntax_xref_rows(
    occurrences_table: pa.Table,
    nodes_table: pa.Table,
) -> list[dict[str, object]]:
    if occurrences_table.num_rows == 0 or nodes_table.num_rows == 0:
        return []
    indexes = _build_syntax_node_indexes(nodes_table)
    occurrences_by_path: dict[str, list[dict[str, object]]] = {}
    for row in iter_rows(occurrences_table):
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        occurrences_by_path.setdefault(rel_path, []).append(row)

    rows: list[dict[str, object]] = []
    for (rel_path, producer), index in indexes.items():
        occ_rows = occurrences_by_path.get(rel_path)
        if not occ_rows:
            continue
        for occ in occ_rows:
            occ_row = {
                "occ_start_byte": occ.get("start_byte"),
                "occ_end_byte": occ.get("end_byte"),
                "occ_start_line": occ.get("start_line"),
                "occ_start_col": occ.get("start_col"),
                "occ_end_line": occ.get("end_line"),
                "occ_end_col": occ.get("end_col"),
            }
            syntax_node_id, match_kind, candidate_count = _match_occurrence_to_node(
                index,
                rel_path,
                occ_row,
            )
            rows.append(
                {
                    "repo": occ.get("repo"),
                    "commit": occ.get("commit"),
                    "rel_path": rel_path,
                    "producer": producer,
                    "scip_symbol": occ.get("scip_symbol"),
                    "scip_occurrence_id": _stable_occurrence_id(
                        {
                            **occ_row,
                            "rel_path": rel_path,
                            "scip_symbol": occ.get("scip_symbol"),
                        }
                    ),
                    "occ_start_byte": occ_row["occ_start_byte"],
                    "occ_end_byte": occ_row["occ_end_byte"],
                    "occ_start_line": occ_row["occ_start_line"],
                    "occ_start_col": occ_row["occ_start_col"],
                    "occ_end_line": occ_row["occ_end_line"],
                    "occ_end_col": occ_row["occ_end_col"],
                    "syntax_node_id": syntax_node_id,
                    "match_kind": match_kind,
                    "candidate_count": candidate_count,
                }
            )
    return rows


def scip_resolution__frames(
    q__core__scip_occurrences: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__syntax_defs_resolved: InferableTabularInput,
) -> ScipResolutionFrames:
    """Build base SCIP resolution frames.

    Returns
    -------
    ScipResolutionFrames
        Frames for SCIP symbol and occurrence xref tables.
    """
    created_at = datetime.now(tz=UTC).replace(tzinfo=None)
    occurrences = _occurrences_table(q__core__scip_occurrences)
    symbol_info = _symbol_info_table(q__core__scip_symbol_information)
    goids = _goids_table(q__core__goids)
    anchors = _definition_anchors_table(q__core__syntax_defs_resolved)
    symbol_goid_xref = _symbol_goid_xref_table(
        occurrences=occurrences,
        goids=goids,
        anchors=anchors,
        created_at=created_at,
    )
    occurrence_span_xref = _occurrence_span_xref_table(
        occurrences=occurrences,
        symbol_info=symbol_info,
        symbol_goid_xref=symbol_goid_xref,
        created_at=created_at,
    )
    return ScipResolutionFrames(
        symbol_goid_xref=symbol_goid_xref,
        occurrence_span_xref=occurrence_span_xref,
    )


def scip_resolution__symbol_goid_xref__base(
    scip_resolution__frames: ScipResolutionFrames,
) -> pa.Table:
    """Return rows for core.scip_symbol_goid_xref.

    Returns
    -------
    pa.Table
        Arrow reader for core.scip_symbol_goid_xref.
    """
    table = dedupe_table_for_table(
        SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
        scip_resolution__frames.symbol_goid_xref,
    )
    if table.num_rows == 0:
        return _empty_reader_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    return align_table_to_contract(
        SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
        table,
        target_name=SCIP_RESOLUTION_TARGET_NAME,
        reporter=emit_alignment_report,
    )


def scip_resolution__occurrence_span_xref__base(
    scip_resolution__frames: ScipResolutionFrames,
) -> pa.Table:
    """Return rows for core.scip_occurrence_span_xref.

    Returns
    -------
    pa.Table
        Arrow reader for core.scip_occurrence_span_xref.
    """
    table = dedupe_table_for_table(
        SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
        scip_resolution__frames.occurrence_span_xref,
    )
    if table.num_rows == 0:
        return _empty_reader_for_output_table(SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY)
    return align_table_to_contract(
        SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
        table,
        target_name=SCIP_RESOLUTION_TARGET_NAME,
        reporter=emit_alignment_report,
    )


def scip_resolution__occurrence_syntax_xref__base(
    scip_resolution__frames: ScipResolutionFrames,
    q__core__syntax_nodes: InferableTabularInput,
) -> pa.Table:
    """Return rows for core.scip_occurrence_syntax_xref.

    Returns
    -------
    pa.Table
        Arrow reader for core.scip_occurrence_syntax_xref.
    """
    occurrences_table = scip_resolution__frames.occurrence_span_xref
    nodes_table = tabular_to_scoped_table(
        q__core__syntax_nodes,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    rows = _occurrence_syntax_xref_rows(occurrences_table, nodes_table)
    if not rows:
        return _empty_reader_for_output_table(SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY)
    table = pa.Table.from_pylist(rows)
    table = dedupe_table_for_table(SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY, table)
    return align_table_to_contract(
        SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY,
        table,
        target_name=SCIP_RESOLUTION_TARGET_NAME,
        reporter=emit_alignment_report,
    )


_MODULE = sys.modules[__name__]
_SCIP_RESOLUTION_TABLE_CONTEXTS = (
    TableTargetTableContext(
        table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
        base_node="scip_resolution__symbol_goid_xref__base",
        node_name="scip_resolution__symbol_goid_xref",
    ),
    TableTargetTableContext(
        table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
        base_node="scip_resolution__occurrence_span_xref__base",
        node_name="scip_resolution__occurrence_span_xref",
    ),
    TableTargetTableContext(
        table_key=SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY,
        base_node="scip_resolution__occurrence_syntax_xref__base",
        node_name="scip_resolution__occurrence_syntax_xref",
    ),
)
_SCIP_RESOLUTION_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=SCIP_RESOLUTION_TARGET_NAME,
        tables=(),
        table_materializations_node="scip_resolution__table_materializations",
        anchor_node_name="t__scip_resolution",
        save_spec_factory=RelationTableSaveSpec,
        default_input_type=InferableTabularInput,
    ),
    table_contexts=_SCIP_RESOLUTION_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_SCIP_RESOLUTION_TABLE_TARGET_SPEC)
scip_resolution__symbol_goid_xref = _MODULE.scip_resolution__symbol_goid_xref
scip_resolution__occurrence_span_xref = _MODULE.scip_resolution__occurrence_span_xref
scip_resolution__occurrence_syntax_xref = _MODULE.scip_resolution__occurrence_syntax_xref
scip_resolution__table_materializations = _MODULE.scip_resolution__table_materializations
t__scip_resolution = _MODULE.t__scip_resolution


__all__ = [
    "SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY",
    "SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY",
    "SCIP_RESOLUTION_TARGET_NAME",
    "SCIP_SYMBOL_GOID_XREF_TABLE_KEY",
    "ScipResolutionFrames",
    "scip_resolution__frames",
    "scip_resolution__occurrence_span_xref",
    "scip_resolution__occurrence_syntax_xref",
    "scip_resolution__symbol_goid_xref",
    "scip_resolution__table_materializations",
    "t__scip_resolution",
]
