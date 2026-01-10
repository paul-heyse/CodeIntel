"""SCIP resolution tables for deterministic symbol/GOID stitching."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

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
from codeintel.build.hamilton.transforms.ingestion_normalize import scoped_table_for_ingest
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinOptions,
    ArrowJoinSpec,
    arrow_join_tables,
    normalize_table_for_join,
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
from codeintel.build.tabular.finalize_ops import (
    FinalizeResult,
    finalize_join_keys,
    record_join_precheck_errors,
)
from codeintel.build.tabular.frames import JoinStrategy
from codeintel.build.tabular.kernels import hash_struct_goid
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.conversion import empty_table_from_schema
from codeintel.core.columnar.dedupe_ops import stable_dedupe_for_context
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.iter import iter_array_values
from codeintel.core.columnar.join_safe import join_safe_projection
from codeintel.core.columnar.kernels import SortKey, case_when, stable_sort_table
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.columnar.schema_ops import concat_tables_unified
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS
from codeintel.core.schemas.primitives import (
    resolve_canonical_sort_keys,
    resolve_join_safe_columns,
)

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
SCIP_OCCURRENCES_TABLE_KEY = "core.scip_occurrences"
SCIP_SYMBOL_INFO_TABLE_KEY = "core.scip_symbol_information"
GOIDS_TABLE_KEY = "core.goids"
SYNTAX_DEFS_TABLE_KEY = "core.syntax_defs"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
_INTERNAL_PLAN_TABLE_KEY = "internal.plan_materialize"

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
_MATCH_PRIORITY_COLUMN = "match_priority"
_MATCH_PRIORITY_BY_KIND = {
    _MATCH_KIND_BYTE_SPAN: 4,
    _MATCH_KIND_LINE_COL: 3,
    _MATCH_KIND_LINE_SPAN: 2,
    _MATCH_KIND_LINE_START: 1,
}
_SYMBOL_GOID_XREF_KEY_COLUMNS = ("repo", "commit", "scip_symbol")
_SYMBOL_GOID_XREF_TIE_BREAKERS = (
    "goid_h128",
    "def_rel_path",
    "def_start_line",
    "def_start_col",
    "def_end_line",
    "def_end_col",
)

_JOIN_STRING_KEYS = {"repo", "commit", "rel_path", "scip_symbol"}
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


def _resolve_ingest_execution_ctx(env: BuildEnv | None) -> ExecutionContext:
    if env is not None:
        resolved = resolve_columnar_context(env.execution_context)
        if resolved is not None:
            return resolved
    fallback: ExecutionContext | None = None
    return resolve_execution_context(fallback)


_OCCURRENCE_ID_COLUMNS = (
    "rel_path",
    "scip_symbol",
    "occ_start_line",
    "occ_start_col",
    "occ_end_line",
    "occ_end_col",
    "occ_start_byte",
    "occ_end_byte",
)


def _join_casts(keys: Sequence[str]) -> dict[str, str]:
    casts: dict[str, str] = {}
    for key in keys:
        if key in _JOIN_STRING_KEYS:
            casts[key] = "string"
        elif key in _JOIN_INT_KEYS:
            casts[key] = "int64"
    return casts


def _cast_table_columns(
    table: pa.Table,
    *,
    casts: dict[str, str],
) -> pa.Table:
    if not casts or table.num_rows == 0:
        return table
    arrays = []
    names = list(table.column_names)
    for name in names:
        column = table[name]
        if name in casts:
            arrays.append(cast_array(column, pa.type_for_alias(casts[name]), safe=False))
        else:
            arrays.append(column)
    return pa.Table.from_arrays(arrays, names=names)


def _precheck_join_table(
    table: pa.Table,
    *,
    table_key: str | None,
    join_keys: Sequence[str],
) -> pa.Table:
    if table.num_rows == 0 or not join_keys:
        return table
    result = finalize_join_keys(
        table,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        result,
        table_key=table_key,
        target_name=SCIP_RESOLUTION_TARGET_NAME,
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


def _join_safe_allowlist(table_key: str | None) -> tuple[str, ...]:
    if table_key is None:
        return ()
    schema = get_schema_service().get_table_schema(table_key)
    return resolve_join_safe_columns(schema)


def _normalize_join_input(
    table: pa.Table,
    *,
    table_key: str | None,
) -> pa.Table:
    normalized = normalize_table_for_join(
        table,
        enforce_join_safe=False,
    )
    return join_safe_projection(
        normalized,
        allowed_columns=_join_safe_allowlist(table_key),
    )


def _hash_join_tables(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: _JoinSpec,
    how: JoinStrategy = "left",
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
    left_checked = _normalize_join_input(
        left_checked,
        table_key=spec.left_table_key,
    )
    right_checked = _normalize_join_input(
        right_checked,
        table_key=spec.right_table_key,
    )
    left_checked = _cast_table_columns(left_checked, casts=_join_casts(spec.left_keys))
    right_checked = _cast_table_columns(right_checked, casts=_join_casts(spec.right_keys))
    right_output = [
        name for name in right_checked.column_names if name not in left_checked.column_names
    ]
    right_keep = list(dict.fromkeys([*spec.right_keys, *right_output]))
    right_selected = right_checked.select(right_keep)
    joined = arrow_join_tables(
        left_checked,
        right_selected,
        spec=ArrowJoinSpec(
            left_on=spec.left_keys,
            right_on=spec.right_keys,
            how=how,
            coalesce_keys=True,
        ),
        options=ArrowJoinOptions(normalize_inputs=False),
    )
    sort_keys: list[SortKey] = [(key, "ascending") for key in spec.left_keys]
    return stable_sort_table(joined, sort_keys=sort_keys) if sort_keys else joined


@dataclass(frozen=True)
class ScipResolutionFrames:
    """Derived frames for SCIP resolution outputs."""

    symbol_goid_xref: pa.Table
    occurrence_span_xref: pa.Table


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


def _empty_plan_for_output_table(table_key: str) -> Plan:
    try:
        empty = empty_table_for_table(table_key)
    except KeyError:
        table_schema = OUTPUT_TABLE_SCHEMAS.get(table_key)
        if table_schema is None:
            raise
        arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
        empty = empty_table_from_schema(arrow_schema)
    return _plan_from_table(empty, table_key=table_key)


def _empty_table_for_output_table(table_key: str) -> pa.Table:
    table_schema = OUTPUT_TABLE_SCHEMAS.get(table_key)
    if table_schema is None:
        msg = f"Missing output schema for {table_key}"
        raise KeyError(msg)
    arrow_schema = arrow_contract_for_table_schema(table_schema=table_schema)
    return empty_table_from_schema(arrow_schema)


def _canonical_sort_keys_for_table(
    table_key: str,
    columns: Sequence[str],
) -> tuple[SortKey, ...] | None:
    schema = get_schema_service().get_table_schema(table_key)
    keys = resolve_canonical_sort_keys(schema)
    if not keys:
        return None
    available = set(columns)
    return tuple((key, "ascending") for key in keys if key in available)


def _plan_from_table(
    table: pa.Table,
    *,
    table_key: str | None,
) -> Plan:
    plan = Plan.table(table)
    if table_key is None:
        return plan
    sort_keys = _canonical_sort_keys_for_table(table_key, table.column_names)
    if sort_keys:
        return plan.order_by(sort_keys=list(sort_keys))
    return plan


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
    table = scoped_table_for_ingest(
        symbol_info,
        table_key=SCIP_SYMBOL_INFO_TABLE_KEY,
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
    table = scoped_table_for_ingest(
        goids,
        table_key=GOIDS_TABLE_KEY,
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


def _definition_anchors_table(
    defs: InferableTabularInput,
    goids: InferableTabularInput,
    *,
    execution_ctx: ExecutionContext | None = None,
) -> pa.Table:
    defs_table = scoped_table_for_ingest(
        defs,
        table_key=SYNTAX_DEFS_TABLE_KEY,
        columns=[
            "repo",
            "commit",
            "rel_path",
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
    goids_table = scoped_table_for_ingest(
        goids,
        table_key=GOIDS_TABLE_KEY,
        columns=[
            "repo",
            "commit",
            "rel_path",
            "start_line",
            "end_line",
            "goid_h128",
        ],
        scope=None,
        require_scope_columns=False,
    )
    defs_table = _cast_int32(defs_table, ["start_line", "start_col", "end_line", "end_col"])
    goids_table = _cast_int32(goids_table, ["start_line", "end_line"])
    if defs_table.num_rows == 0 or goids_table.num_rows == 0:
        goid_type = _goid_type_for_table(goids_table)
        return _append_null_column(defs_table, "goid_h128", goid_type)
    join_keys = ["repo", "commit", "rel_path", "start_line", "end_line"]
    joined = _hash_join_tables(
        defs_table,
        goids_table,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            left_table_key=SYNTAX_DEFS_TABLE_KEY,
            right_table_key=GOIDS_TABLE_KEY,
        ),
    )
    if "goid_h128" not in joined.column_names:
        return joined
    return safe_filter(joined, is_valid_mask(joined["goid_h128"]))


def _occurrences_table(occurrences: InferableTabularInput) -> pa.Table:
    table = scoped_table_for_ingest(
        occurrences,
        table_key=SCIP_OCCURRENCES_TABLE_KEY,
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
    grouped = grouped_rollup_table(
        goids,
        spec=GroupedRollupSpec(
            keys=("rel_path", "start_line"),
            aggregates=(("goid_h128", "min", None, "goid_h128"),),
            order_by=(("rel_path", "ascending"), ("start_line", "ascending")),
        ),
    )
    return grouped.select(["rel_path", "start_line", "goid_h128"])


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


@dataclass(frozen=True, slots=True)
class _AnchorGoidMatchRequest:
    definitions: pa.Table
    anchors: pa.Table
    join_keys: Sequence[str]
    match_kind: str
    confidence: float
    execution_ctx: ExecutionContext | None = None


def _anchor_goid_matches(request: _AnchorGoidMatchRequest) -> tuple[pa.Table, pa.Table]:
    definitions = request.definitions
    anchors = request.anchors
    join_keys = request.join_keys
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
                joined = _hash_join_tables(
                    left_valid,
                    anchors_valid,
                    spec=_JoinSpec(
                        left_keys=list(join_keys),
                        right_keys=list(join_keys),
                    ),
                )
                matched_mask = is_valid_mask(joined["goid_h128"])
                matched = safe_filter(joined, matched_mask)
                matched = _attach_match_metadata(
                    matched,
                    kind=request.match_kind,
                    confidence=request.confidence,
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
    *,
    execution_ctx: ExecutionContext | None = None,
) -> tuple[pa.Table, pa.Table]:
    anchor_cols = ["rel_path", "start_byte", "end_byte", "goid_h128"]
    if not set(anchor_cols).issubset(anchors.column_names):
        return definitions.slice(0, 0), definitions
    return _anchor_goid_matches(
        _AnchorGoidMatchRequest(
            definitions=definitions,
            anchors=anchors.select(anchor_cols),
            join_keys=["rel_path", "start_byte", "end_byte"],
            match_kind=_MATCH_KIND_BYTE_SPAN,
            confidence=_MATCH_CONFIDENCE_BYTE_SPAN,
            execution_ctx=execution_ctx,
        )
    )


def _line_col_goid_matches(
    definitions: pa.Table,
    anchors: pa.Table,
    *,
    execution_ctx: ExecutionContext | None = None,
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
        _AnchorGoidMatchRequest(
            definitions=definitions,
            anchors=anchors.select(anchor_cols),
            join_keys=["rel_path", "start_line", "start_col", "end_line", "end_col"],
            match_kind=_MATCH_KIND_LINE_COL,
            confidence=_MATCH_CONFIDENCE_LINE_COL,
            execution_ctx=execution_ctx,
        )
    )


def _strict_goid_matches(
    definitions: pa.Table,
    goids: pa.Table,
    *,
    execution_ctx: ExecutionContext | None = None,
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
    join_keys = ["rel_path", "start_line", "end_line"]
    strict_join = _hash_join_tables(
        left,
        goids,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            right_table_key=GOIDS_TABLE_KEY,
        ),
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


def _fallback_goid_matches(
    definitions: pa.Table,
    goids: pa.Table,
    *,
    execution_ctx: ExecutionContext | None = None,
) -> list[pa.Table]:
    if definitions.num_rows == 0:
        return []
    fallback_left = _drop_columns_if_present(
        definitions,
        ["goid_h128", "match_kind", "match_confidence"],
    )
    goids_by_line = _goids_by_start_line(goids)
    fallback_keys = ["rel_path", "start_line"]
    fallback_join = _hash_join_tables(
        fallback_left,
        goids_by_line,
        spec=_JoinSpec(
            left_keys=fallback_keys,
            right_keys=fallback_keys,
            right_table_key=GOIDS_TABLE_KEY,
        ),
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
    for value in iter_array_values(table.column("match_kind")):
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    summary = " ".join(f"{key}={count}" for key, count in sorted(counts.items()))
    LOG.info(
        "scip_symbol_goid_xref match coverage total=%d %s",
        table.num_rows,
        summary,
    )


def _sort_keys_for_columns(
    table: pa.Table,
    columns: Sequence[str],
    *,
    order: Literal["ascending", "descending"],
) -> tuple[SortKey, ...]:
    available = set(table.column_names)
    return tuple((name, order) for name in columns if name in available)


def _match_priority_array(
    match_kind: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    cases = [
        (
            equal_mask(match_kind, pa.scalar(kind, type=match_kind.type)),
            pa.scalar(priority),
        )
        for kind, priority in _MATCH_PRIORITY_BY_KIND.items()
    ]
    return case_when(cases, else_=pa.scalar(0))


def _append_match_priority(table: pa.Table) -> pa.Table:
    if table.num_rows == 0 or "match_kind" not in table.column_names:
        return table
    match_kind = table["match_kind"]
    priorities = _match_priority_array(match_kind)
    return table.append_column(_MATCH_PRIORITY_COLUMN, priorities)


def _dedupe_symbol_goid_xref(
    table: pa.Table,
    *,
    execution_ctx: ExecutionContext | None,
) -> pa.Table:
    if table.num_rows == 0:
        return table
    with_priority = _append_match_priority(table)
    order_by = (
        *_sort_keys_for_columns(
            with_priority,
            (_MATCH_PRIORITY_COLUMN,),
            order="descending",
        ),
        *_sort_keys_for_columns(
            with_priority,
            ("match_confidence",),
            order="descending",
        ),
    )
    tie_breakers = _sort_keys_for_columns(
        with_priority,
        _SYMBOL_GOID_XREF_TIE_BREAKERS,
        order="ascending",
    )
    deduped = stable_dedupe_for_context(
        with_priority,
        key_columns=_SYMBOL_GOID_XREF_KEY_COLUMNS,
        order_by=order_by,
        tie_breakers=tie_breakers,
        ctx=execution_ctx,
    )
    if _MATCH_PRIORITY_COLUMN in deduped.column_names:
        return deduped.drop_columns([_MATCH_PRIORITY_COLUMN])
    return deduped


def _symbol_goid_xref_table(
    *,
    occurrences: pa.Table,
    goids: pa.Table,
    anchors: pa.Table,
    created_at: datetime,
    execution_ctx: ExecutionContext | None = None,
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
        byte_matched, remaining = _byte_span_goid_matches(
            remaining,
            anchors,
            execution_ctx=execution_ctx,
        )
        if byte_matched.num_rows != 0:
            matched_tables.append(byte_matched)
        line_col_matched, remaining = _line_col_goid_matches(
            remaining,
            anchors,
            execution_ctx=execution_ctx,
        )
        if line_col_matched.num_rows != 0:
            matched_tables.append(line_col_matched)
    fallback_tables: list[pa.Table] = []
    if goids.num_rows != 0:
        strict_matched, strict_missing = _strict_goid_matches(
            remaining,
            goids,
            execution_ctx=execution_ctx,
        )
        if strict_matched.num_rows != 0:
            matched_tables.append(strict_matched)
        fallback_tables = _fallback_goid_matches(
            strict_missing,
            goids,
            execution_ctx=execution_ctx,
        )
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
    execution_ctx: ExecutionContext | None = None,
) -> pa.Table:
    goid_lookup_source = _dedupe_symbol_goid_xref(
        symbol_goid_xref,
        execution_ctx=execution_ctx,
    )
    goid_lookup = goid_lookup_source.select(
        [
            "repo",
            "commit",
            "scip_symbol",
            "goid_h128",
        ]
    )
    join_keys = ["repo", "commit", "scip_symbol"]
    base = _hash_join_tables(
        occurrences,
        symbol_info,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            left_table_key=SCIP_OCCURRENCES_TABLE_KEY,
            right_table_key=SCIP_SYMBOL_INFO_TABLE_KEY,
        ),
    )
    base = _hash_join_tables(
        base,
        goid_lookup,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            right_table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
        ),
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


_OCCURRENCE_MATCH_BASE_COLUMNS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "scip_occurrence_id",
)
_OCCURRENCE_MATCH_KEYS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "producer",
    "scip_occurrence_id",
)
_OCCURRENCE_MATCH_SORT_KEYS: tuple[SortKey, ...] = tuple(
    (key, "ascending") for key in _OCCURRENCE_MATCH_KEYS
)
_OCCURRENCE_SYNTAX_OUTPUT_COLUMNS: tuple[str, ...] = (
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
_OCCURRENCE_BYTE_JOIN_KEYS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "occ_start_byte",
    "occ_end_byte",
)
_SYNTAX_BYTE_JOIN_KEYS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "start_byte",
    "end_byte",
)
_OCCURRENCE_LINE_JOIN_KEYS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "occ_start_line",
    "occ_start_col",
    "occ_end_line",
    "occ_end_col",
)
_SYNTAX_LINE_JOIN_KEYS: tuple[str, ...] = (
    "repo",
    "commit",
    "rel_path",
    "start_line",
    "start_col",
    "end_line",
    "end_col",
)


def _precheck_join_keys(
    table: pa.Table,
    *,
    join_keys: Sequence[str],
    table_key: str | None,
) -> pa.Table:
    if table.num_rows == 0 or not join_keys:
        return table
    result = finalize_join_keys(
        table,
        required_non_null=join_keys,
        key_fields=join_keys,
        stage="join_precheck",
    )
    record_join_precheck_errors(
        result,
        table_key=table_key,
        target_name=SCIP_RESOLUTION_TARGET_NAME,
        join_keys=join_keys,
    )
    _log_join_precheck_errors(result, table_key=table_key, join_keys=join_keys)
    return result.good


def _unique_columns(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in columns:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _unique_marker_name(columns: Sequence[str]) -> str:
    base = "__right_marker"
    if base not in columns:
        return base
    counter = 1
    while f"{base}_{counter}" in columns:
        counter += 1
    return f"{base}_{counter}"


def _occurrence_syntax_occurrences_table(
    occurrences_table: pa.Table,
) -> pa.Table:
    selected = occurrences_table.select(
        [
            "repo",
            "commit",
            "rel_path",
            "scip_symbol",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "start_byte",
            "end_byte",
        ]
    )
    occurrences = _rename_columns(
        selected,
        {
            "start_line": "occ_start_line",
            "start_col": "occ_start_col",
            "end_line": "occ_end_line",
            "end_col": "occ_end_col",
            "start_byte": "occ_start_byte",
            "end_byte": "occ_end_byte",
        },
    )
    hashed = hash_struct_goid(occurrences, columns=_OCCURRENCE_ID_COLUMNS)
    occurrence_ids = cast_array(hashed, pa.string(), safe=False)
    return occurrences.append_column("scip_occurrence_id", occurrence_ids)


def _occurrence_syntax_producers_table(
    nodes_table: pa.Table,
    *,
    execution_ctx: ExecutionContext | None = None,
) -> pa.Table:
    join_keys = ("repo", "commit", "rel_path", "producer")
    checked = _precheck_join_keys(
        nodes_table,
        join_keys=join_keys,
        table_key=SYNTAX_NODES_TABLE_KEY,
    )
    projected = checked.select(list(join_keys))
    sort_keys: list[SortKey] = [(name, "ascending") for name in join_keys]
    return grouped_rollup_table(
        projected,
        spec=GroupedRollupSpec(keys=join_keys, aggregates=(), order_by=sort_keys),
        ctx=execution_ctx,
    )


def _occurrence_syntax_pairs_table(
    occurrences: pa.Table,
    producers: pa.Table,
) -> pa.Table:
    join_keys = ("repo", "commit", "rel_path")
    return _hash_join_tables(
        occurrences,
        producers,
        spec=_JoinSpec(
            left_keys=join_keys,
            right_keys=join_keys,
            left_table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
            right_table_key=SYNTAX_NODES_TABLE_KEY,
        ),
        how="inner",
    )


@dataclass(frozen=True, slots=True)
class _OccurrenceSyntaxMatchRequest:
    occurrences: pa.Table
    syntax_nodes: pa.Table
    left_keys: Sequence[str]
    right_keys: Sequence[str]
    match_kind: str
    execution_ctx: ExecutionContext | None = None


def _occurrence_syntax_match_table(request: _OccurrenceSyntaxMatchRequest) -> pa.Table:
    occurrences = request.occurrences
    syntax_nodes = request.syntax_nodes
    if occurrences.num_rows == 0 or syntax_nodes.num_rows == 0:
        return pa.table({})
    left_checked = _precheck_join_keys(
        occurrences,
        join_keys=request.left_keys,
        table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
    )
    right_checked = _precheck_join_keys(
        syntax_nodes,
        join_keys=request.right_keys,
        table_key=SYNTAX_NODES_TABLE_KEY,
    )
    left_checked = _normalize_join_input(
        left_checked,
        table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
    )
    right_checked = _normalize_join_input(
        right_checked,
        table_key=SYNTAX_NODES_TABLE_KEY,
    )
    left_columns = _unique_columns([*_OCCURRENCE_MATCH_BASE_COLUMNS, *request.left_keys])
    right_columns = _unique_columns([*request.right_keys, "producer", "node_id"])
    left_selected = left_checked.select(left_columns)
    right_selected = right_checked.select(right_columns)
    joined = arrow_join_tables(
        left_selected,
        right_selected,
        spec=ArrowJoinSpec(
            left_on=request.left_keys,
            right_on=request.right_keys,
            how="inner",
            coalesce_keys=True,
        ),
        options=ArrowJoinOptions(normalize_inputs=False),
    )
    sort_keys: list[SortKey] = [
        (name, "ascending") for name in (*_OCCURRENCE_MATCH_BASE_COLUMNS, "producer")
    ]
    grouped = grouped_rollup_table(
        joined,
        spec=GroupedRollupSpec(
            keys=(*_OCCURRENCE_MATCH_BASE_COLUMNS, "producer"),
            aggregates=(
                ("node_id", "min", None, "syntax_node_id"),
                ("node_id", "count", None, "candidate_count"),
            ),
            order_by=sort_keys,
        ),
        ctx=request.execution_ctx,
    )
    match_kind = constant_array(request.match_kind, grouped.num_rows)
    return pa.Table.from_arrays(
        [
            grouped["repo"],
            grouped["commit"],
            grouped["rel_path"],
            grouped["producer"],
            grouped["scip_occurrence_id"],
            grouped["syntax_node_id"],
            match_kind,
            grouped["candidate_count"],
        ],
        names=[
            "repo",
            "commit",
            "rel_path",
            "producer",
            "scip_occurrence_id",
            "syntax_node_id",
            "match_kind",
            "candidate_count",
        ],
    )


def _occurrence_syntax_left_anti(
    left: pa.Table,
    right: pa.Table,
    *,
    execution_ctx: ExecutionContext | None = None,
) -> pa.Table:
    if left.num_rows == 0 or right.num_rows == 0:
        return left
    left_checked = _precheck_join_keys(
        left,
        join_keys=_OCCURRENCE_MATCH_KEYS,
        table_key=None,
    )
    right_checked = _precheck_join_keys(
        right,
        join_keys=_OCCURRENCE_MATCH_KEYS,
        table_key=None,
    )
    left_checked = _normalize_join_input(
        left_checked,
        table_key=None,
    )
    right_checked = _normalize_join_input(
        right_checked,
        table_key=None,
    )
    marker_name = _unique_marker_name(right_checked.column_names)
    right_marker = right_checked.append_column(
        marker_name,
        constant_array(value=True, length=right_checked.num_rows),
    )
    right_selected = right_marker.select([*_OCCURRENCE_MATCH_KEYS, marker_name])
    joined = arrow_join_tables(
        left_checked,
        right_selected,
        spec=ArrowJoinSpec(
            left_on=_OCCURRENCE_MATCH_KEYS,
            right_on=_OCCURRENCE_MATCH_KEYS,
            how="left",
            coalesce_keys=True,
        ),
        options=ArrowJoinOptions(normalize_inputs=False),
    )
    filtered = safe_filter_expr(joined, E.is_null(marker_name))
    if marker_name in filtered.column_names:
        filtered = filtered.drop_columns([marker_name])
    return filtered


def _empty_occurrence_match_table(pairs: pa.Table) -> pa.Table:
    arrays = []
    names = []
    for name in _OCCURRENCE_MATCH_KEYS:
        arrays.append(pa.array([], type=pairs.schema.field(name).type))
        names.append(name)
    arrays.append(pa.array([], type=pa.string()))
    names.append("syntax_node_id")
    arrays.append(pa.array([], type=pa.string()))
    names.append("match_kind")
    arrays.append(pa.array([], type=pa.int64()))
    names.append("candidate_count")
    return pa.Table.from_arrays(arrays, names=names)


def scip_resolution__frames(
    env: BuildEnv,
    q__core__scip_occurrences: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__syntax_defs: InferableTabularInput,
) -> ScipResolutionFrames:
    """Build base SCIP resolution frames.

    Parameters
    ----------
    env
        Build environment providing execution context defaults.
    q__core__scip_occurrences
        SCIP occurrence rows for resolution.
    q__core__scip_symbol_information
        SCIP symbol metadata for resolution.
    q__core__goids
        GOID rows for resolution.
    q__core__syntax_defs
        Syntax definition rows for anchor resolution.

    Returns
    -------
    ScipResolutionFrames
        Frames for SCIP symbol and occurrence xref tables.
    """
    created_at = datetime.now(tz=UTC).replace(tzinfo=None)
    execution_ctx = _resolve_ingest_execution_ctx(env)
    occurrences = _occurrences_table(q__core__scip_occurrences)
    symbol_info = _symbol_info_table(q__core__scip_symbol_information)
    goids = _goids_table(q__core__goids)
    anchors = _definition_anchors_table(
        q__core__syntax_defs,
        q__core__goids,
        execution_ctx=execution_ctx,
    )
    symbol_goid_xref = _symbol_goid_xref_table(
        occurrences=occurrences,
        goids=goids,
        anchors=anchors,
        created_at=created_at,
        execution_ctx=execution_ctx,
    )
    occurrence_span_xref = _occurrence_span_xref_table(
        occurrences=occurrences,
        symbol_info=symbol_info,
        symbol_goid_xref=symbol_goid_xref,
        created_at=created_at,
        execution_ctx=execution_ctx,
    )
    return ScipResolutionFrames(
        symbol_goid_xref=symbol_goid_xref,
        occurrence_span_xref=occurrence_span_xref,
    )


def scip_resolution__symbol_goid_xref__base(
    _env: BuildEnv,
    scip_resolution__frames: ScipResolutionFrames,
) -> Plan:
    """Return rows for core.scip_symbol_goid_xref.

    Parameters
    ----------
    _env
        Build environment with snapshot metadata.
    scip_resolution__frames
        Resolved SCIP frames.

    Returns
    -------
    Plan
        Plan for core.scip_symbol_goid_xref.
    """
    table = scip_resolution__frames.symbol_goid_xref
    if table.num_rows == 0:
        return _empty_plan_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    return _plan_from_table(table, table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY)


def scip_resolution__occurrence_span_xref__base(
    _env: BuildEnv,
    scip_resolution__frames: ScipResolutionFrames,
) -> Plan:
    """Return rows for core.scip_occurrence_span_xref.

    Parameters
    ----------
    _env
        Build environment with snapshot metadata.
    scip_resolution__frames
        Resolved SCIP frames.

    Returns
    -------
    Plan
        Plan for core.scip_occurrence_span_xref.
    """
    table = scip_resolution__frames.occurrence_span_xref
    if table.num_rows == 0:
        return _empty_plan_for_output_table(SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY)
    return _plan_from_table(table, table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY)


def scip_resolution__occurrence_syntax_xref__base(
    env: BuildEnv,
    scip_resolution__frames: ScipResolutionFrames,
    q__core__syntax_nodes: InferableTabularInput,
) -> Plan:
    """Return rows for core.scip_occurrence_syntax_xref.

    Parameters
    ----------
    env
        Build environment with snapshot metadata.
    scip_resolution__frames
        Resolved SCIP frames.
    q__core__syntax_nodes
        Syntax node inputs for occurrence resolution.

    Returns
    -------
    Plan
        Plan for core.scip_occurrence_syntax_xref.
    """
    execution_ctx = _resolve_ingest_execution_ctx(env)
    occurrences_table = scip_resolution__frames.occurrence_span_xref
    nodes_table = scoped_table_for_ingest(
        q__core__syntax_nodes,
        table_key=SYNTAX_NODES_TABLE_KEY,
        columns=None,
        scope=None,
        require_scope_columns=False,
    )
    if occurrences_table.num_rows == 0 or nodes_table.num_rows == 0:
        return _empty_plan_for_output_table(SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY)

    occurrences = _occurrence_syntax_occurrences_table(occurrences_table)
    producers = _occurrence_syntax_producers_table(nodes_table, execution_ctx=execution_ctx)
    if occurrences.num_rows == 0 or producers.num_rows == 0:
        return _empty_plan_for_output_table(SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY)

    pairs = _occurrence_syntax_pairs_table(occurrences, producers)
    if pairs.num_rows == 0:
        return _empty_plan_for_output_table(SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY)

    byte_matches = _occurrence_syntax_match_table(
        _OccurrenceSyntaxMatchRequest(
            occurrences=occurrences,
            syntax_nodes=nodes_table,
            left_keys=_OCCURRENCE_BYTE_JOIN_KEYS,
            right_keys=_SYNTAX_BYTE_JOIN_KEYS,
            match_kind="EXACT",
            execution_ctx=execution_ctx,
        )
    )
    line_matches = _occurrence_syntax_match_table(
        _OccurrenceSyntaxMatchRequest(
            occurrences=occurrences,
            syntax_nodes=nodes_table,
            left_keys=_OCCURRENCE_LINE_JOIN_KEYS,
            right_keys=_SYNTAX_LINE_JOIN_KEYS,
            match_kind="EXACT",
            execution_ctx=execution_ctx,
        )
    )
    if byte_matches.num_rows != 0:
        line_matches = _occurrence_syntax_left_anti(
            line_matches,
            byte_matches,
            execution_ctx=execution_ctx,
        )
    if byte_matches.num_rows == 0:
        matches = line_matches
    elif line_matches.num_rows == 0:
        matches = byte_matches
    else:
        matches = concat_tables_unified([byte_matches, line_matches])

    pairs = _precheck_join_keys(
        pairs,
        join_keys=_OCCURRENCE_MATCH_KEYS,
        table_key=None,
    )
    matches = _precheck_join_keys(
        matches,
        join_keys=_OCCURRENCE_MATCH_KEYS,
        table_key=None,
    )
    pairs = _normalize_join_input(
        pairs,
        table_key=None,
    )
    matches = _normalize_join_input(
        matches,
        table_key=None,
    )
    joined = build_table_plan(table=pairs).hash_join(
        right=build_table_plan(table=matches),
        spec=HashJoinSpec(
            left_keys=list(_OCCURRENCE_MATCH_KEYS),
            right_keys=list(_OCCURRENCE_MATCH_KEYS),
            how="left outer",
            left_output=list(pairs.column_names),
            right_output=["syntax_node_id", "match_kind", "candidate_count"],
        ),
    )
    project: dict[str, Expression] = {
        "repo": E.field("repo"),
        "commit": E.field("commit"),
        "rel_path": E.field("rel_path"),
        "producer": E.field("producer"),
        "scip_symbol": E.field("scip_symbol"),
        "scip_occurrence_id": E.field("scip_occurrence_id"),
        "occ_start_byte": E.field("occ_start_byte"),
        "occ_end_byte": E.field("occ_end_byte"),
        "occ_start_line": E.field("occ_start_line"),
        "occ_start_col": E.field("occ_start_col"),
        "occ_end_line": E.field("occ_end_line"),
        "occ_end_col": E.field("occ_end_col"),
        "syntax_node_id": E.field("syntax_node_id"),
        "match_kind": E.coalesce([E.field("match_kind"), E.scalar("NONE")]),
        "candidate_count": E.coalesce([E.field("candidate_count"), E.scalar(0)]),
    }
    projected = joined.project(project)
    sort_keys = _canonical_sort_keys_for_table(
        SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY,
        _OCCURRENCE_SYNTAX_OUTPUT_COLUMNS,
    )
    if sort_keys is None:
        ordered = projected.order_by(sort_keys=list(_OCCURRENCE_MATCH_SORT_KEYS))
    else:
        ordered = projected.order_by(sort_keys=list(sort_keys))
    return ordered.project({name: E.field(name) for name in _OCCURRENCE_SYNTAX_OUTPUT_COLUMNS})


_MODULE = sys.modules[__name__]


def _scip_resolution_save_spec(table_key: str) -> RelationTableSaveSpec:
    return RelationTableSaveSpec(
        table_key=table_key,
        ingest_finalize=True,
    )


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
        save_spec_factory=_scip_resolution_save_spec,
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
