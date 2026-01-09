"""SCIP plane CPG nodes and edges."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_TARGET_NAME
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
)
from codeintel.build.hamilton.native.graphs.cpg2.edge_helpers import (
    finalize_cpg_edge_rows,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal, cpg_node_id
from codeintel.build.tabular.arrow_ops import normalize_table_for_join
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter_expr
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_expr, is_valid_mask
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.build.tabular.finalize_ops import (
    FinalizeDedupe,
    FinalizeResult,
    FinalizeSpec,
    finalize_join_keys,
    finalize_table,
    record_join_precheck_errors,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.intervals.span_resolver import SpanResolver

LOG = logging.getLogger(__name__)

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
SCIP_EXTERNAL_SYMBOLS_TABLE_KEY = "core.scip_external_symbols"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"

OccurrenceSpanKey = tuple[object, object, object, object, object, object, object, object]


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


_SCIP_JOIN_KEYS = ("repo", "commit", "symbol")


def _scip_symbol_joined_table(
    symbols: pa.Table,
    *,
    table_key: str,
    left_output: Sequence[str],
) -> pa.Table:
    normalized = canonicalize_for_table(symbols, table_key=table_key)
    normalized = normalize_table_for_join(normalized)
    normalized = _precheck_join_table(
        normalized,
        table_key=table_key,
        join_keys=_SCIP_JOIN_KEYS,
    )
    anchors = build_anchor_map(
        normalized,
        table_key=table_key,
        pk_columns=identity_keys(table_key),
        include_source_pk_json=True,
    )
    anchors = normalize_table_for_join(anchors)
    anchors = _precheck_join_table(
        anchors,
        table_key=None,
        join_keys=_SCIP_JOIN_KEYS,
    )
    left_exprs = _join_key_exprs()
    for column in left_output:
        if column in left_exprs:
            continue
        left_exprs[column] = E.field(column)
    right_exprs = {
        **_join_key_exprs(),
        "cpg_node_id": E.field("cpg_node_id"),
        "source_pk_json": E.field("source_pk_json"),
    }
    left_plan = Plan.table(normalized).project(left_exprs)
    right_plan = Plan.table(anchors).project(right_exprs)
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=list(_SCIP_JOIN_KEYS),
            right_keys=list(_SCIP_JOIN_KEYS),
            how="left outer",
            left_output=list(left_exprs.keys()),
            right_output=["cpg_node_id", "source_pk_json"],
        ),
    )
    joined = joined.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("symbol", "ascending"),
        ],
    )
    return materialize_plan(joined, use_threads=True)


def _join_key_exprs() -> dict[str, Expression]:
    return {
        "repo": E.cast(E.field("repo"), "string"),
        "commit": E.cast(E.field("commit"), "string"),
        "symbol": E.cast(E.field("symbol"), "string"),
    }


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
            ),
        )
    record_join_precheck_errors(
        result,
        table_key=table_key,
        target_name=CPG_TARGET_NAME,
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
    joined = _scip_symbol_joined_table(
        symbols,
        table_key=SCIP_SYMBOLS_TABLE_KEY,
        left_output=list(_SCIP_JOIN_KEYS),
    )
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "SCIP_SYMBOL",
            "source_table_key": SCIP_SYMBOLS_TABLE_KEY,
            "rel_path": None,
            "start_byte": None,
            "end_byte": None,
            "extras": None,
            "extras_kv": None,
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
            "extras",
            "extras_kv",
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


def cpg2_nodes__scip_external_symbols(
    symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG nodes from external SCIP symbols.

    Returns
    -------
    pyarrow.Table
        CPG node table for external SCIP symbols.
    """
    required = {"repo", "commit", "symbol"}
    if not required.issubset(set(symbols.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    desired_output = [
        *_SCIP_JOIN_KEYS,
        "package_manager",
        "package_name",
        "package_version",
    ]
    left_output = [column for column in desired_output if column in symbols.column_names]
    joined = _scip_symbol_joined_table(
        symbols,
        table_key=SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
        left_output=left_output,
    )
    extras_kv = _external_symbol_extras_kv(joined)
    joined = _upsert_column(joined, "extras_kv", extras_kv)
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "SCIP_SYMBOL_EXTERNAL",
            "source_table_key": SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
            "rel_path": None,
            "start_byte": None,
            "end_byte": None,
            "extras": None,
            "extras_kv": None,
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
            "extras",
            "extras_kv",
        ]
    )
    filtered = _filter_valid_nodes(selected)
    if diagnostics is not None:
        diagnostics["scip_external_symbols"] = ScipNodeDiagnostics(
            total_rows=selected.num_rows,
            resolved_rows=filtered.num_rows,
            dropped_rows=selected.num_rows - filtered.num_rows,
        )
    return filtered


def cpg2_edges__scip_occurrences(
    occ_syntax: pa.Table,
    occ_span: pa.Table,
    scip_symbols: pa.Table,
    scip_external_symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG edges from SCIP occurrence-to-syntax matches.

    Returns
    -------
    pyarrow.Table
        CPG edges for SCIP occurrence bindings.
    """
    internal_keys = _symbol_key_set(scip_symbols)
    external_keys = _symbol_key_set(scip_external_symbols)
    joined = _occurrence_roles(occ_syntax, occ_span)
    rows = [
        row
        for row in (
            _scip_occurrence_edge_row(
                source=row,
                internal_keys=internal_keys,
                external_keys=external_keys,
            )
            for row in table_rows(joined)
        )
        if row is not None
    ]
    table = finalize_cpg_edge_rows(rows)
    filtered = _filter_valid_edges(table)
    if diagnostics is not None:
        diagnostics["scip_occurrences"] = ScipOccurrenceDiagnostics(
            total_edges=table.num_rows,
            resolved_edges=filtered.num_rows,
            dropped_edges=table.num_rows - filtered.num_rows,
        )
    return filtered


def _scip_occurrence_edge_row(
    *,
    source: dict[str, object],
    internal_keys: set[tuple[str, str, str]],
    external_keys: set[tuple[str, str, str]],
) -> dict[str, object] | None:
    if source.get("syntax_node_id") is None:
        return None
    syntax_pk = {
        "repo": source.get("repo"),
        "commit": source.get("commit"),
        "rel_path": source.get("rel_path"),
        "producer": source.get("producer"),
        "node_id": source.get("syntax_node_id"),
    }
    symbol_pk = {
        "repo": source.get("repo"),
        "commit": source.get("commit"),
        "symbol": source.get("scip_symbol"),
    }
    dst_table_key, is_external = _symbol_table_key(
        source.get("repo"),
        source.get("commit"),
        source.get("scip_symbol"),
        internal_keys=internal_keys,
        external_keys=external_keys,
    )
    edge_kind = _edge_kind_for_occurrence(source)
    extras_values = {
        "scip_occurrence_id": source.get("scip_occurrence_id"),
        "match_kind": source.get("match_kind"),
        "candidate_count": source.get("candidate_count"),
        "scip_roles": source.get("scip_roles"),
        "span_match_kind": source.get("span_match_kind"),
        "span_candidate_count": source.get("span_candidate_count"),
    }
    if is_external:
        extras_values["symbol_origin"] = "external"
    extras_kv = extras_kv_from_mapping(extras_values)
    ordinal = cpg_edge_ordinal(
        "core.scip_occurrence_syntax_xref",
        {"scip_occurrence_id": source.get("scip_occurrence_id")},
    )
    return {
        "repo": source.get("repo"),
        "commit": source.get("commit"),
        "src_cpg_node_id": cpg_node_id(SYNTAX_NODES_TABLE_KEY, syntax_pk),
        "dst_cpg_node_id": cpg_node_id(dst_table_key, symbol_pk),
        "edge_kind": edge_kind,
        "edge_layer": "SYMBOL",
        "rel_path": source.get("rel_path"),
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _edge_kind_for_occurrence(source: dict[str, object]) -> str:
    if _flag_is_true(source.get("is_definition")):
        return "DEFINES"
    if _flag_is_true(source.get("is_import")):
        return "IMPORTS"
    if _flag_is_true(source.get("is_write")):
        return "WRITES"
    return "REFERS_TO"


def _flag_is_true(value: object) -> bool:
    return bool(value) if value is not None else False


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
        return pa.Table.from_pydict({})
    span_index = _occurrence_span_index(table_rows(occ_span))
    resolvers = _occurrence_role_resolvers(occ_span)
    joined_rows = _occurrence_joined_rows(syntax_rows, span_index)
    _apply_occurrence_resolvers(joined_rows, resolvers)
    return _table_from_rows(joined_rows)


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


def _table_from_rows(rows: list[dict[str, object]]) -> pa.Table:
    if not rows:
        return pa.Table.from_pydict({})
    column_names = sorted({name for row in rows for name in row})
    data: dict[str, list[object]] = {name: [] for name in column_names}
    for row in rows:
        for name in column_names:
            data[name].append(row.get(name))
    return pa.Table.from_pydict(data)


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

    def _mask(target: pa.Table) -> pa.Array | pa.ChunkedArray:
        return and_kleene(
            is_valid_mask(target.column("src_cpg_node_id")),
            is_valid_mask(target.column("dst_cpg_node_id")),
        )

    expr = is_valid_expr("src_cpg_node_id") & is_valid_expr("dst_cpg_node_id")
    return safe_filter_expr(table, expr, fallback_mask=_mask)


def _filter_valid_nodes(table: pa.Table) -> pa.Table:
    if "cpg_node_id" not in table.column_names:
        return table

    def _mask(target: pa.Table) -> pa.Array | pa.ChunkedArray:
        return is_valid_mask(target.column("cpg_node_id"))

    return safe_filter_expr(table, is_valid_expr("cpg_node_id"), fallback_mask=_mask)


def _symbol_key_set(table: pa.Table) -> set[tuple[str, str, str]]:
    required = {"repo", "commit", "symbol"}
    if not required.issubset(set(table.column_names)):
        return set()
    keys: set[tuple[str, str, str]] = set()
    for row in table_rows(table):
        repo = row.get("repo")
        commit = row.get("commit")
        symbol = row.get("symbol")
        if isinstance(repo, str) and isinstance(commit, str) and isinstance(symbol, str):
            keys.add((repo, commit, symbol))
    return keys


def _symbol_table_key(
    repo: object,
    commit: object,
    symbol: object,
    *,
    internal_keys: set[tuple[str, str, str]],
    external_keys: set[tuple[str, str, str]],
) -> tuple[str, bool]:
    if not isinstance(repo, str) or not isinstance(commit, str) or not isinstance(symbol, str):
        return SCIP_SYMBOLS_TABLE_KEY, False
    key = (repo, commit, symbol)
    if key in internal_keys:
        return SCIP_SYMBOLS_TABLE_KEY, False
    if key in external_keys:
        return SCIP_EXTERNAL_SYMBOLS_TABLE_KEY, True
    return SCIP_EXTERNAL_SYMBOLS_TABLE_KEY, True


def _external_symbol_extras_kv(table: pa.Table) -> pa.Array:
    extras_values: list[dict[str, str] | None] = []
    for row in table_rows(table):
        values = {
            "package_manager": row.get("package_manager"),
            "package_name": row.get("package_name"),
            "package_version": row.get("package_version"),
        }
        extras_values.append(extras_kv_from_mapping(values))
    return pa.array(extras_values, type=pa.map_(pa.string(), pa.string()))


def _upsert_column(
    table: pa.Table,
    name: str,
    values: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index == -1:
        return table.append_column(name, values)
    return table.set_column(index, name, values)


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
    "cpg2_nodes__scip_external_symbols",
    "cpg2_nodes__scip_symbols",
]
