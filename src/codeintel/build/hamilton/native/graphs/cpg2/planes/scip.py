"""SCIP plane CPG nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal, cpg_node_id
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    JoinFilterClause,
    arrow_join_tables,
    build_join_options,
    join_filter_expr,
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_expr, is_valid_mask
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.serialization.payload import encode_payload

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"

OccurrenceSpanKey = tuple[object, object, object, object, object, object, object, object]

_EXPR_TYPE = getattr(pc, "Expression", None)


@dataclass(frozen=True)
class ScipNodeDiagnostics:
    """Diagnostics for SCIP symbol nodes."""

    total_rows: int
    resolved_rows: int
    dropped_rows: int


@dataclass(frozen=True)
class ScipOccurrenceDiagnostics:
    """Diagnostics for SCIP occurrence edges."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class _OccurrenceRolePayload:
    scip_roles: int | None
    is_definition: bool | None
    is_reference: bool | None
    is_import: bool | None
    is_write: bool | None
    is_read: bool | None


def cpg2_nodes__scip_symbols(
    symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG nodes from SCIP symbol metadata.

    Returns
    -------
    pyarrow.Table
        CPG node table for SCIP symbols.
    """
    required = {"repo", "commit", "symbol"}
    if not required.issubset(set(symbols.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    normalized = canonicalize_for_table(symbols, table_key=SCIP_SYMBOLS_TABLE_KEY)
    normalized = normalize_table_for_join(normalized)
    anchors = build_anchor_map(
        normalized,
        table_key=SCIP_SYMBOLS_TABLE_KEY,
        pk_columns=identity_keys(SCIP_SYMBOLS_TABLE_KEY),
        include_source_pk_json=True,
    )
    anchors = normalize_table_for_join(anchors)
    join_spec = ArrowJoinSpec(on=["repo", "commit", "symbol"], how="left")
    filter_expr = join_filter_expr(
        left=normalized,
        right=anchors,
        spec=join_spec,
        clause=JoinFilterClause(
            field="cpg_node_id",
            predicate=is_valid_expr,
            side="right",
        ),
    )
    join_options = build_join_options(
        normalized,
        anchors,
        filter_expression=filter_expr,
        normalize_inputs=False,
    )
    joined = arrow_join_tables(
        normalized,
        anchors,
        spec=join_spec,
        options=join_options,
    )
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "SCIP_SYMBOL",
            "source_table_key": SCIP_SYMBOLS_TABLE_KEY,
            "rel_path": None,
            "start_byte": None,
            "end_byte": None,
            "extras_json": None,
        },
    )
    selected = joined.select(
        [
            "repo",
            "commit",
            "cpg_node_id",
            "node_kind",
            "source_table_key",
            "source_pk_json",
            "rel_path",
            "start_byte",
            "end_byte",
            "extras_json",
        ]
    )
    filtered = _filter_valid_nodes(selected)
    if diagnostics is not None:
        diagnostics["scip_symbols"] = ScipNodeDiagnostics(
            total_rows=selected.num_rows,
            resolved_rows=filtered.num_rows,
            dropped_rows=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__scip_occurrences(
    occ_syntax: pa.Table,
    occ_span: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges from SCIP occurrence-to-syntax matches.

    Returns
    -------
    pyarrow.Table
        CPG edges for SCIP occurrence bindings.
    """
    joined = _occurrence_roles(occ_syntax, occ_span)
    rows: list[dict[str, object]] = []
    for row in table_rows(joined):
        if row.get("syntax_node_id") is None:
            continue
        syntax_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
            "node_id": row.get("syntax_node_id"),
        }
        symbol_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "symbol": row.get("scip_symbol"),
        }
        is_def = bool(row.get("is_definition")) if row.get("is_definition") is not None else False
        is_import = bool(row.get("is_import")) if row.get("is_import") is not None else False
        is_write = bool(row.get("is_write")) if row.get("is_write") is not None else False
        is_read = bool(row.get("is_read")) if row.get("is_read") is not None else False
        edge_kind = "REFERS_TO"
        if is_def:
            edge_kind = "DEFINES"
        elif is_import:
            edge_kind = "IMPORTS"
        elif is_write:
            edge_kind = "WRITES"
        elif is_read:
            edge_kind = "REFERS_TO"
        extras_values = {
            "scip_occurrence_id": row.get("scip_occurrence_id"),
            "match_kind": row.get("match_kind"),
            "candidate_count": row.get("candidate_count"),
            "scip_roles": row.get("scip_roles"),
            "span_match_kind": row.get("span_match_kind"),
            "span_candidate_count": row.get("span_candidate_count"),
        }
        ordinal = cpg_edge_ordinal(
            "core.scip_occurrence_syntax_xref",
            {"scip_occurrence_id": row.get("scip_occurrence_id")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(SYNTAX_NODES_TABLE_KEY, syntax_pk),
                "dst_cpg_node_id": cpg_node_id(SCIP_SYMBOLS_TABLE_KEY, symbol_pk),
                "edge_kind": edge_kind,
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _payload_bytes(extras_values),
            }
        )
    table, _ = table_for_rows(CPG_EDGES_TABLE_KEY, rows)
    filtered = _filter_valid_edges(table)
    if diagnostics is not None:
        diagnostics["scip_occurrences"] = ScipOccurrenceDiagnostics(
            total_edges=table.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=table.num_rows - filtered.num_rows,
        )
    return filtered


def _occurrence_role_resolvers(
    span_frame: pa.Table,
) -> dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]]:
    resolvers: dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]] = {}
    for row in table_rows(span_frame):
        rel_path = row.get("rel_path")
        scip_symbol = row.get("scip_symbol")
        if not isinstance(rel_path, str) or not isinstance(scip_symbol, str):
            continue
        start_line = row.get("occ_start_line", row.get("start_line"))
        end_line = row.get("occ_end_line", row.get("end_line"))
        if not isinstance(start_line, int):
            continue
        end_line_value = end_line if isinstance(end_line, int) else start_line
        resolver = resolvers.get((rel_path, scip_symbol))
        if resolver is None:
            resolver = SpanResolver.for_lines(path_normalizer=lambda value: value)
            resolvers[rel_path, scip_symbol] = resolver
        resolver.add_span(
            rel_path,
            start_line,
            end_line_value,
            _OccurrenceRolePayload(
                scip_roles=_coerce_int(row.get("scip_roles", row.get("roles"))),
                is_definition=_coerce_bool(row.get("is_definition")),
                is_reference=_coerce_bool(row.get("is_reference")),
                is_import=_coerce_bool(row.get("is_import")),
                is_write=_coerce_bool(row.get("is_write")),
                is_read=_coerce_bool(row.get("is_read")),
            ),
        )
    return resolvers


def _occurrence_roles(
    occ_syntax: pa.Table,
    occ_span: pa.Table,
) -> pa.Table:
    syntax_rows = table_rows(occ_syntax)
    if not syntax_rows:
        return pa.Table.from_pylist([])
    span_index = _occurrence_span_index(table_rows(occ_span))
    resolvers = _occurrence_role_resolvers(occ_span)
    joined_rows = _occurrence_joined_rows(syntax_rows, span_index)
    _apply_occurrence_resolvers(joined_rows, resolvers)
    return pa.Table.from_pylist(joined_rows)


def _occurrence_span_index(
    span_rows: list[dict[str, object]],
) -> dict[OccurrenceSpanKey, dict[str, object]]:
    index: dict[OccurrenceSpanKey, dict[str, object]] = {}
    for row in span_rows:
        key = (
            row.get("repo"),
            row.get("commit"),
            row.get("rel_path"),
            row.get("scip_symbol"),
            row.get("start_line"),
            row.get("start_col"),
            row.get("end_line"),
            row.get("end_col"),
        )
        index[key] = row
    return index


def _occurrence_joined_rows(
    syntax_rows: list[dict[str, object]],
    span_index: dict[OccurrenceSpanKey, dict[str, object]],
) -> list[dict[str, object]]:
    joined_rows: list[dict[str, object]] = []
    for row in syntax_rows:
        key = (
            row.get("repo"),
            row.get("commit"),
            row.get("rel_path"),
            row.get("scip_symbol"),
            row.get("occ_start_line"),
            row.get("occ_start_col"),
            row.get("occ_end_line"),
            row.get("occ_end_col"),
        )
        span_row = span_index.get(key)
        joined = dict(row)
        if span_row is not None:
            joined["scip_roles"] = span_row.get("roles")
            joined["is_definition"] = span_row.get("is_definition")
            joined["is_reference"] = span_row.get("is_reference")
            joined["is_import"] = span_row.get("is_import")
            joined["is_write"] = span_row.get("is_write")
            joined["is_read"] = span_row.get("is_read")
        else:
            joined["scip_roles"] = None
            joined["is_definition"] = None
            joined["is_reference"] = None
            joined["is_import"] = None
            joined["is_write"] = None
            joined["is_read"] = None
        joined["span_match_kind"] = None
        joined["span_candidate_count"] = None
        joined_rows.append(joined)
    return joined_rows


def _apply_occurrence_resolvers(
    joined_rows: list[dict[str, object]],
    resolvers: dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]],
) -> None:
    for joined in joined_rows:
        if joined.get("scip_roles") is not None:
            continue
        rel_path = joined.get("rel_path")
        scip_symbol = joined.get("scip_symbol")
        if not isinstance(rel_path, str) or not isinstance(scip_symbol, str):
            continue
        resolver = resolvers.get((rel_path, scip_symbol))
        start_line = joined.get("occ_start_line")
        if resolver is None or not isinstance(start_line, int):
            continue
        end_line = joined.get("occ_end_line")
        end_line_value = end_line if isinstance(end_line, int) else start_line
        match = resolver.resolve(rel_path, start_line, end_line_value)
        if match.match_kind == "NONE" or match.payload is None:
            continue
        payload = match.payload
        joined["scip_roles"] = payload.scip_roles
        joined["is_definition"] = payload.is_definition
        joined["is_reference"] = payload.is_reference
        joined["is_import"] = payload.is_import
        joined["is_write"] = payload.is_write
        joined["is_read"] = payload.is_read
        joined["span_match_kind"] = match.match_kind
        joined["span_candidate_count"] = match.candidate_count


def _filter_valid_edges(table: pa.Table) -> pa.Table:
    required = {"src_cpg_node_id", "dst_cpg_node_id"}
    if not required.issubset(set(table.column_names)):
        return table
    if _EXPR_TYPE is not None:
        try:
            expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
            return safe_filter(table, expr)
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            TypeError,
            ValueError,
        ):
            pass
    mask = and_kleene(
        is_valid_mask(table.column("src_cpg_node_id")),
        is_valid_mask(table.column("dst_cpg_node_id")),
    )
    return safe_filter(table, mask)


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    if "cpg_node_id" not in table.column_names:
        return table
    if _EXPR_TYPE is not None:
        try:
            return safe_filter(table, is_valid_expr("cpg_node_id"))
        except (
            pa.ArrowInvalid,
            pa.ArrowNotImplementedError,
            pa.ArrowTypeError,
            TypeError,
            ValueError,
        ):
            pass
    mask = is_valid_mask(table.column("cpg_node_id"))
    return safe_filter(table, mask)


def _payload_bytes(values: dict[str, object]) -> bytes:
    encoded = encode_payload(values)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


__all__ = [
    "ScipNodeDiagnostics",
    "ScipOccurrenceDiagnostics",
    "cpg2_edges__scip_occurrences",
    "cpg2_nodes__scip_symbols",
]
