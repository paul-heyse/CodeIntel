"""Syntax-plane CPG node and edge assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import ensure_table_columns, rename_table_columns
from codeintel.build.hamilton.native.graphs.cpg2.anchors import (
    build_anchor_map,
    canonicalize_for_table,
    identity_keys,
    lookup_keys,
)
from codeintel.build.tabular.arrow_ops import ArrowJoinSpec, arrow_join_tables
from codeintel.build.tabular.compute_columns import append_constant_columns
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import and_kleene, is_valid_mask
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.serialization.payload import encode_payload

SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"


@dataclass(frozen=True)
class SyntaxEdgeDiagnostics:
    """Diagnostics for syntax edge resolution."""

    total_edges: int
    resolved_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class SyntaxNodeDiagnostics:
    """Diagnostics for syntax node resolution."""

    total_nodes: int
    resolved_nodes: int


def _syntax_anchor_map(syntax_nodes: pa.Table, *, include_source_pk_json: bool = True) -> pa.Table:
    """Return anchor map for syntax nodes.

    Returns
    -------
    pyarrow.Table
        Anchor map containing syntax node identifiers.
    """
    normalized = canonicalize_for_table(syntax_nodes, table_key=SYNTAX_NODES_TABLE_KEY)
    return build_anchor_map(
        normalized,
        table_key=SYNTAX_NODES_TABLE_KEY,
        pk_columns=identity_keys(SYNTAX_NODES_TABLE_KEY),
        include_source_pk_json=include_source_pk_json,
    )


def cpg2_nodes__syntax_nodes(
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG syntax nodes from core.syntax_nodes.

    Returns
    -------
    pyarrow.Table
        CPG node table for syntax nodes.
    """
    if syntax_nodes.num_rows == 0:
        return _empty_node_table()
    required = set(identity_keys(SYNTAX_NODES_TABLE_KEY))
    if not required.issubset(set(syntax_nodes.column_names)):
        return _empty_node_table()
    normalized = _encode_extras_json(syntax_nodes, column_name="extras_json")
    anchor_map = _syntax_anchor_map(normalized, include_source_pk_json=True)
    joined = arrow_join_tables(
        normalized,
        anchor_map,
        spec=ArrowJoinSpec(on=["repo", "commit", "rel_path", "producer", "node_id"], how="left"),
    )
    joined = _encode_extras_json(joined, column_name="extras_json")
    joined = append_constant_columns(
        joined,
        {
            "node_kind": "SYNTAX_NODE",
            "source_table_key": SYNTAX_NODES_TABLE_KEY,
        },
    )
    selected = ensure_table_columns(
        joined,
        (
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
        ),
    )
    if diagnostics is not None:
        resolved = _count_valid(selected, "cpg_node_id")
        diagnostics["syntax_nodes"] = SyntaxNodeDiagnostics(
            total_nodes=selected.num_rows,
            resolved_nodes=resolved,
        )
    return selected


def cpg2_edges__syntax_edges(
    syntax_edges: pa.Table,
    syntax_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build CPG syntax edges from core.syntax_edges.

    Returns
    -------
    pyarrow.Table
        CPG edge table for syntax edges.
    """
    if syntax_edges.num_rows == 0:
        return _empty_edge_table()
    join_keys = lookup_keys(SYNTAX_NODES_TABLE_KEY, "full")
    required = set(join_keys) | {"parent_node_id", "child_node_id"}
    if not required.issubset(set(syntax_edges.column_names)):
        return _empty_edge_table()
    anchor_map = _syntax_anchor_map(syntax_nodes, include_source_pk_json=False)
    normalized_edges = canonicalize_for_table(syntax_edges, table_key="core.syntax_edges")
    parent_left_on = ["parent_node_id" if column == "node_id" else column for column in join_keys]
    child_left_on = ["child_node_id" if column == "node_id" else column for column in join_keys]
    parent_join = arrow_join_tables(
        normalized_edges,
        anchor_map,
        spec=ArrowJoinSpec(
            left_on=parent_left_on,
            right_on=list(join_keys),
            how="left",
        ),
    )
    child_anchor = rename_table_columns(anchor_map, {"cpg_node_id": "cpg_node_id_child"})
    child_join = arrow_join_tables(
        parent_join,
        child_anchor,
        spec=ArrowJoinSpec(
            left_on=child_left_on,
            right_on=list(join_keys),
            how="left",
            right_suffix="_child",
        ),
    )
    child_join = rename_table_columns(child_join, {"cpg_node_id": "src_cpg_node_id"})
    if "cpg_node_id_child" in child_join.column_names:
        child_join = rename_table_columns(child_join, {"cpg_node_id_child": "dst_cpg_node_id"})
    child_join = append_constant_columns(
        child_join,
        {
            "edge_kind": "AST",
            "edge_layer": "SYNTAX",
        },
    )
    selected = ensure_table_columns(
        child_join,
        (
            "repo",
            "commit",
            "src_cpg_node_id",
            "dst_cpg_node_id",
            "edge_kind",
            "edge_layer",
            "rel_path",
            "child_ordinal",
            "extras_json",
        ),
    )
    selected = rename_table_columns(selected, {"child_ordinal": "ordinal"})
    mask = and_kleene(
        is_valid_mask(selected.column("src_cpg_node_id")),
        is_valid_mask(selected.column("dst_cpg_node_id")),
    )
    filtered = safe_filter(selected, mask)
    if diagnostics is not None:
        resolved = filtered.num_rows
        diagnostics["syntax_edges"] = SyntaxEdgeDiagnostics(
            total_edges=selected.num_rows,
            resolved_edges=resolved,
            dropped_edges=selected.num_rows - resolved,
        )
    return filtered


def _count_valid(table: pa.Table, column: str) -> int:
    if column not in table.column_names:
        return 0
    result = pc.call_function("sum", [is_valid_mask(table[column])])
    if isinstance(result, pa.Scalar):
        return int(result.as_py() or 0)
    return int(result or 0)


def _empty_node_table() -> pa.Table:
    return empty_table_for_table(CPG_NODES_TABLE_KEY)


def _empty_edge_table() -> pa.Table:
    return empty_table_for_table(CPG_EDGES_TABLE_KEY)


_CPG_NODE_COLUMNS = (
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
)

_CPG_EDGE_COLUMNS = (
    "repo",
    "commit",
    "src_cpg_node_id",
    "dst_cpg_node_id",
    "edge_kind",
    "edge_layer",
    "rel_path",
    "ordinal",
    "extras_json",
)


def _encode_extras_json(table: pa.Table, *, column_name: str) -> pa.Table:
    if column_name not in table.column_names:
        return table
    encoded = [_encode_optional_payload(value) for value in table[column_name].to_pylist()]
    index = table.schema.get_field_index(column_name)
    return table.set_column(index, column_name, pa.array(encoded, type=pa.binary()))


def _encode_optional_payload(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return encode_payload(bytes(value))
    if isinstance(value, Mapping):
        return encode_payload(dict(value))
    if isinstance(value, (str, int, float, bool)):
        return encode_payload(value)
    return None


__all__ = [
    "SyntaxEdgeDiagnostics",
    "SyntaxNodeDiagnostics",
    "cpg2_edges__syntax_edges",
    "cpg2_nodes__syntax_nodes",
]
