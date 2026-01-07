"""SCIP resolution tables for deterministic symbol/GOID stitching."""

from __future__ import annotations

import hashlib
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
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec,
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
from codeintel.build.tabular.compute_helpers import cast_array
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    bit_wise_and,
    equal_mask,
    is_valid_mask,
    not_equal_mask,
)
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.output_registry import OUTPUT_TABLE_SCHEMAS

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    DagCatalog,
    TargetRunRecord,
    InferableTabularInput,
)

SCIP_RESOLUTION_TARGET_NAME = "scip_resolution"
SCIP_SYMBOL_GOID_XREF_TABLE_KEY = "core.scip_symbol_goid_xref"
SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY = "core.scip_occurrence_span_xref"
SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY = "core.scip_occurrence_syntax_xref"

_ROLE_DEFINITION = 0x1
_ROLE_IMPORT = 0x2
_ROLE_WRITE = 0x4
_ROLE_READ = 0x8


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


def _symbol_info_table(symbol_info: InferableTabularInput) -> pa.Table:
    table = tabular_to_arrow_table(symbol_info).select(
        [
            "repo",
            "commit",
            "symbol",
            "enclosing_symbol",
        ]
    )
    return _rename_columns(table, {"symbol": "scip_symbol"})


def _goids_table(goids: InferableTabularInput) -> pa.Table:
    table = tabular_to_arrow_table(goids).select(
        [
            "goid_h128",
            "rel_path",
            "start_line",
            "end_line",
        ]
    )
    table = _cast_int32(table, ["start_line", "end_line"])
    if table.num_rows == 0:
        return table
    mask = and_kleene(
        is_valid_mask(table["start_line"]),
        is_valid_mask(table["end_line"]),
    )
    return table.filter(mask)


def _occurrences_table(occurrences: InferableTabularInput) -> pa.Table:
    table = tabular_to_arrow_table(occurrences)
    table = _rename_columns(table, {"symbol": "scip_symbol"})
    return _cast_int32(table, ["start_line", "end_line"])


def _symbol_goid_xref_table(
    *,
    occurrences: pa.Table,
    goids: pa.Table,
    created_at: datetime,
) -> pa.Table:
    if occurrences.num_rows == 0 or goids.num_rows == 0:
        return _empty_table_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    roles = occurrences["roles"] if "roles" in occurrences.column_names else None
    if roles is None:
        return _empty_table_for_output_table(SCIP_SYMBOL_GOID_XREF_TABLE_KEY)
    def_mask = not_equal_mask(
        bit_wise_and(roles, pa.scalar(_ROLE_DEFINITION, type=roles.type)),
        pa.scalar(0, type=roles.type),
    )
    definitions = occurrences.filter(def_mask)
    join_spec = ArrowJoinSpec(on=["rel_path", "start_line", "end_line"], how="left", validate="m:1")
    join_options = build_join_options(definitions, goids)
    joined = arrow_join_tables(definitions, goids, spec=join_spec, options=join_options)
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
            "is_definition",
            "is_reference",
            "is_import",
            "is_write",
            "is_read",
            "enclosing_symbol",
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
    symbol_goid_xref = _symbol_goid_xref_table(
        occurrences=occurrences,
        goids=goids,
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
    nodes_table = tabular_to_arrow_table(q__core__syntax_nodes)
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
_SCIP_RESOLUTION_TABLE_TARGET_SPEC = build_multi_table_target_spec(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=SCIP_RESOLUTION_TARGET_NAME,
        tables=(
            MultiTableTargetContext.build_relation_table_spec(
                context=TableTargetTableContext(
                    table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
                    base_node="scip_resolution__symbol_goid_xref__base",
                    node_name="scip_resolution__symbol_goid_xref",
                    input_type=InferableTabularInput,
                ),
            ),
            MultiTableTargetContext.build_relation_table_spec(
                context=TableTargetTableContext(
                    table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
                    base_node="scip_resolution__occurrence_span_xref__base",
                    node_name="scip_resolution__occurrence_span_xref",
                    input_type=InferableTabularInput,
                ),
            ),
            MultiTableTargetContext.build_relation_table_spec(
                context=TableTargetTableContext(
                    table_key=SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY,
                    base_node="scip_resolution__occurrence_syntax_xref__base",
                    node_name="scip_resolution__occurrence_syntax_xref",
                    input_type=InferableTabularInput,
                ),
            ),
        ),
        table_materializations_node="scip_resolution__table_materializations",
        anchor_node_name="t__scip_resolution",
    )
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
