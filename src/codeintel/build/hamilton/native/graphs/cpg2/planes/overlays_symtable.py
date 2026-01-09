"""Symtable overlay CPG edges."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.edge_helpers import (
    finalize_cpg_edge_rows,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal, cpg_node_id
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.core.columnar.rows import empty_table_for_table

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
AST_NODES_TABLE_KEY = "core.ast_nodes"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY = "core.py_sym_unresolved_bindings"
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = "core.py_sym_resolution_edges"


@dataclass(frozen=True)
class OverlayEdgeDiagnostics:
    """Diagnostics for overlay edge resolution."""

    expected_edges: int
    produced_edges: int
    dropped_edges: int


def cpg2_edges__py_sym_scope_edges(
    scope_edges: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build symtable scope edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for scope relationships.
    """
    required = {
        "repo",
        "commit",
        "rel_path",
        "parent_scope_id",
        "child_scope_id",
        "edge_kind",
    }
    if not required.issubset(scope_edges.column_names):
        return _empty_edges()
    rows: list[dict[str, object]] = []
    for row in table_rows(scope_edges):
        parent_scope_id = row.get("parent_scope_id")
        child_scope_id = row.get("child_scope_id")
        if parent_scope_id is None or child_scope_id is None:
            continue
        parent_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": parent_scope_id,
        }
        child_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": child_scope_id,
        }
        extras_kv = extras_kv_from_mapping({"edge_kind": row.get("edge_kind")})
        owns_ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_scope",
            {
                "parent_scope_id": parent_scope_id,
                "child_scope_id": child_scope_id,
                "edge_kind": "OWNS_SCOPE",
            },
        )
        parent_ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_scope",
            {
                "parent_scope_id": parent_scope_id,
                "child_scope_id": child_scope_id,
                "edge_kind": "PARENT_SCOPE",
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(PY_SYM_SCOPES_TABLE_KEY, parent_pk),
                "dst_cpg_node_id": cpg_node_id(PY_SYM_SCOPES_TABLE_KEY, child_pk),
                "edge_kind": "OWNS_SCOPE",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": owns_ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(PY_SYM_SCOPES_TABLE_KEY, child_pk),
                "dst_cpg_node_id": cpg_node_id(PY_SYM_SCOPES_TABLE_KEY, parent_pk),
                "edge_kind": "PARENT_SCOPE",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": parent_ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table = finalize_cpg_edge_rows(rows)
    _record_diagnostics(
        diagnostics,
        "overlay_symtable_scope_edges",
        expected_edges=scope_edges.num_rows * 2,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_sym_namespace_edges(
    namespace_edges: pa.Table,
    bindings: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build symtable namespace edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for namespace bindings.
    """
    edge_rows = _collect_rows(
        namespace_edges,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "scope_id",
            "name",
            "symbol_row_id",
            "child_scope_id",
            "edge_kind",
            "is_ambiguous",
        ),
    )
    binding_rows = _collect_rows(
        bindings,
        columns=("repo", "commit", "rel_path", "binding_id", "scope_id", "name"),
    )
    if not edge_rows or not binding_rows:
        return _empty_edges()
    bindings_by_scope, _ = _build_binding_index(binding_rows)
    rows: list[dict[str, object]] = []
    for row in edge_rows:
        edge = _namespace_edge_row(row, bindings_by_scope)
        if edge:
            rows.append(edge)
    table = finalize_cpg_edge_rows(rows)
    _record_diagnostics(
        diagnostics,
        "overlay_symtable_namespace_edges",
        expected_edges=namespace_edges.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_sym_binding_edges(
    bindings: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build symtable binding declaration edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for scope -> binding declarations.
    """
    required = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "binding_id",
        "binding_kind",
        "name",
    }
    if not required.issubset(bindings.column_names):
        return _empty_edges()
    rows: list[dict[str, object]] = []
    for row in table_rows(bindings):
        scope_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": row.get("scope_id"),
        }
        binding_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "binding_id": row.get("binding_id"),
        }
        extras_values = {
            "binding_kind": row.get("binding_kind"),
            "declared_here": row.get("declared_here"),
            "referenced_here": row.get("referenced_here"),
            "assigned_here": row.get("assigned_here"),
            "annotated_here": row.get("annotated_here"),
            "scoping_class": row.get("scoping_class"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_binding",
            {
                "scope_id": row.get("scope_id"),
                "binding_id": row.get("binding_id"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(PY_SYM_SCOPES_TABLE_KEY, scope_pk),
                "dst_cpg_node_id": cpg_node_id(PY_SYM_BINDINGS_TABLE_KEY, binding_pk),
                "edge_kind": "DECLARES",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table = finalize_cpg_edge_rows(rows)
    _record_diagnostics(
        diagnostics,
        "overlay_symtable_binding_edges",
        expected_edges=bindings.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_sym_resolution_edges(
    resolution_edges: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build symtable resolution edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for binding resolution.
    """
    required = {
        "repo",
        "commit",
        "rel_path",
        "edge_id",
        "src_binding_id",
        "dst_binding_id",
        "kind",
    }
    if not required.issubset(resolution_edges.column_names):
        return _empty_edges()
    rows = [_py_sym_resolution_edge_row(row) for row in table_rows(resolution_edges)]
    table = finalize_cpg_edge_rows(rows)
    _record_diagnostics(
        diagnostics,
        "overlay_symtable_resolution_edges",
        expected_edges=resolution_edges.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_sym_binding_symbol_edges(
    bindings: pa.Table,
    scopes: pa.Table,
    scip_symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build symtable binding -> SCIP symbol edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for binding-symbol linking.
    """
    required_bindings = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "binding_id",
        "binding_kind",
        "name",
    }
    required_scopes = {"repo", "commit", "rel_path", "scope_id", "qualpath"}
    required_symbols = {"repo", "commit", "symbol", "display_name"}
    if (
        not required_bindings.issubset(bindings.column_names)
        or not required_scopes.issubset(scopes.column_names)
        or not required_symbols.issubset(scip_symbols.column_names)
    ):
        return _empty_edges()
    scope_index = _scope_qualname_index(scopes)
    symbol_index = _symbol_display_index(scip_symbols)
    rows = _binding_symbol_edge_rows(bindings, scope_index, symbol_index)
    table = finalize_cpg_edge_rows(rows)
    _record_diagnostics(
        diagnostics,
        "overlay_symtable_binding_symbol_edges",
        expected_edges=bindings.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__ast_binding_edges(
    ast_nodes: pa.Table,
    scopes: pa.Table,
    bindings: pa.Table,
    resolution_edges: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build AST binding edges derived from symtable bindings.

    Returns
    -------
    pyarrow.Table
        CPG edges linking AST nodes to bindings.
    """
    edges, event_count = _ast_binding_edges_to_rows(
        ast_nodes,
        scopes,
        bindings,
        resolution_edges,
    )
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_symtable_ast_binding_edges",
        expected_edges=event_count,
        produced_edges=table.num_rows,
    )
    return table


def _py_sym_resolution_edge_row(row: Mapping[str, object]) -> dict[str, object]:
    src_pk = {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "rel_path": row.get("rel_path"),
        "binding_id": row.get("src_binding_id"),
    }
    dst_pk = {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "rel_path": row.get("rel_path"),
        "binding_id": row.get("dst_binding_id"),
    }
    dst_table_key = (
        PY_SYM_UNRESOLVED_BINDINGS_TABLE_KEY
        if _is_unresolved_binding(row)
        else PY_SYM_BINDINGS_TABLE_KEY
    )
    extras_values = {
        "kind": row.get("kind"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
    }
    extras_kv = extras_kv_from_mapping(extras_values)
    ordinal = cpg_edge_ordinal(
        PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
        {"edge_id": row.get("edge_id")},
    )
    return {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "src_cpg_node_id": cpg_node_id(PY_SYM_BINDINGS_TABLE_KEY, src_pk),
        "dst_cpg_node_id": cpg_node_id(dst_table_key, dst_pk),
        "edge_kind": "RESOLVES_TO",
        "edge_layer": "SYMBOL",
        "rel_path": row.get("rel_path"),
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _is_unresolved_binding(row: Mapping[str, object]) -> bool:
    dst_binding_id = row.get("dst_binding_id")
    if isinstance(dst_binding_id, str) and dst_binding_id.endswith(":unknown"):
        return True
    kind = row.get("kind")
    return isinstance(kind, str) and kind == "UNKNOWN"


def _namespace_edge_row(
    row: Mapping[str, object],
    bindings_by_scope: Mapping[tuple[str, str, str], dict[str, object]],
) -> dict[str, object]:
    rel_path = _coerce_str(row.get("rel_path"))
    scope_id = _coerce_str(row.get("scope_id"))
    name = _coerce_str(row.get("name"))
    child_scope_id = _coerce_str(row.get("child_scope_id"))
    if rel_path is None or scope_id is None or name is None or child_scope_id is None:
        return {}
    binding = bindings_by_scope.get((rel_path, scope_id, name))
    if binding is None:
        return {}
    repo = _coerce_str(binding.get("repo"))
    commit = _coerce_str(binding.get("commit"))
    binding_id = _coerce_str(binding.get("binding_id"))
    if _has_missing(repo, commit, binding_id):
        return {}
    src_cpg_node_id = cpg_node_id(
        PY_SYM_BINDINGS_TABLE_KEY,
        {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "binding_id": binding_id,
        },
    )
    dst_cpg_node_id = cpg_node_id(
        PY_SYM_SCOPES_TABLE_KEY,
        {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "scope_id": child_scope_id,
        },
    )
    extras = {
        "name": name,
        "symbol_row_id": row.get("symbol_row_id"),
        "is_ambiguous": row.get("is_ambiguous"),
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_namespace",
        {
            "binding_id": binding_id,
            "child_scope_id": child_scope_id,
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": _coerce_str(row.get("edge_kind")) or "BINDS_NAMESPACE",
        "edge_layer": "SYMBOL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _scope_qualname_index(scopes: pa.Table) -> dict[tuple[object, object, object, object], str]:
    index: dict[tuple[object, object, object, object], str] = {}
    for row in table_rows(scopes):
        scope_key = (row.get("repo"), row.get("commit"), row.get("rel_path"), row.get("scope_id"))
        qualname = _scope_qualname_from_qualpath(row.get("qualpath"))
        if qualname:
            index[scope_key] = qualname
    return index


def _symbol_display_index(
    scip_symbols: pa.Table,
) -> dict[tuple[object, object, object], list[Mapping[str, object]]]:
    index: dict[tuple[object, object, object], list[Mapping[str, object]]] = {}
    for row in table_rows(scip_symbols):
        key = (row.get("repo"), row.get("commit"), row.get("display_name"))
        index.setdefault(key, []).append(row)
    return index


def _binding_symbol_edge_rows(
    bindings: pa.Table,
    scope_index: Mapping[tuple[object, object, object, object], str],
    symbol_index: Mapping[tuple[object, object, object], Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in table_rows(bindings):
        scope_key = (row.get("repo"), row.get("commit"), row.get("rel_path"), row.get("scope_id"))
        scope_qualname = scope_index.get(scope_key)
        name = _coerce_str(row.get("name"))
        if scope_qualname is None or name is None:
            continue
        binding_qualname = f"{scope_qualname}.{name}"
        symbols = symbol_index.get((row.get("repo"), row.get("commit"), binding_qualname))
        if not symbols:
            continue
        binding_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "binding_id": row.get("binding_id"),
        }
        for symbol in symbols:
            symbol_pk = {
                "repo": symbol.get("repo"),
                "commit": symbol.get("commit"),
                "symbol": symbol.get("symbol"),
            }
            extras_values = {
                "binding_kind": row.get("binding_kind"),
                "match_kind": "qualpath",
            }
            extras_kv = extras_kv_from_mapping(extras_values)
            ordinal = cpg_edge_ordinal(
                "graph.cpg_edges_binding_symbol",
                {"binding_id": row.get("binding_id"), "symbol": symbol.get("symbol")},
            )
            rows.append(
                {
                    "repo": row.get("repo"),
                    "commit": row.get("commit"),
                    "src_cpg_node_id": cpg_node_id(PY_SYM_BINDINGS_TABLE_KEY, binding_pk),
                    "dst_cpg_node_id": cpg_node_id(SCIP_SYMBOLS_TABLE_KEY, symbol_pk),
                    "edge_kind": "BINDS_SYMBOL",
                    "edge_layer": "SYMBOL",
                    "rel_path": row.get("rel_path"),
                    "ordinal": ordinal,
                    "extras": None,
                    "extras_kv": extras_kv,
                }
            )
    return rows


def _ast_binding_edges_to_rows(
    ast_nodes: pa.Table,
    scopes: pa.Table,
    bindings: pa.Table,
    resolution_edges: pa.Table,
) -> tuple[list[dict[str, object]], int]:
    ast_rows = _collect_rows(
        ast_nodes,
        columns=(
            "path",
            "node_type",
            "hash",
            "ctx",
            "identifier",
            "name",
            "imported",
            "asname",
            "start_byte",
            "end_byte",
            "lineno",
        ),
    )
    if not ast_rows:
        return [], 0
    scope_rows = _collect_rows(
        scopes,
        columns=(
            "rel_path",
            "scope_id",
            "lineno",
            "span_start_byte",
            "span_end_byte",
        ),
    )
    binding_rows = _collect_rows(
        bindings,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "binding_id",
            "scope_id",
            "name",
            "binding_kind",
        ),
    )
    resolution_rows = _collect_rows(
        resolution_edges,
        columns=("src_binding_id", "dst_binding_id", "kind", "confidence", "reason"),
    )
    scopes_by_path = _scopes_by_path(scope_rows)
    bindings_by_scope, binding_meta = _build_binding_index(binding_rows)
    resolution_map = _build_resolution_map(resolution_rows)
    edges: list[dict[str, object]] = []
    event_count = 0
    for row in ast_rows:
        event = _ast_event_row(row)
        if event is None:
            continue
        event_count += 1
        context = _ast_binding_context_for_event(
            event,
            scopes_by_path=scopes_by_path,
            bindings_by_scope=bindings_by_scope,
            binding_meta=binding_meta,
            resolution_map=resolution_map,
        )
        if context is None:
            continue
        edge = _ast_binding_edge_row(event=event, context=context)
        if edge:
            edges.append(edge)
    return edges, event_count


def _binding_payload_from_row(
    row: Mapping[str, object],
) -> tuple[tuple[str, str, str], str, dict[str, object]] | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    binding_id = _coerce_str(row.get("binding_id"))
    scope_id = _coerce_str(row.get("scope_id"))
    name = _coerce_str(row.get("name"))
    binding_kind = _coerce_str(row.get("binding_kind"))
    if _has_missing(repo, commit, rel_path, binding_id, scope_id, name, binding_kind):
        return None
    rel_path_value = cast("str", rel_path)
    scope_id_value = cast("str", scope_id)
    name_value = cast("str", name)
    scope_key = (rel_path_value, scope_id_value, name_value)
    binding_id_value = cast("str", binding_id)
    payload: dict[str, object] = {
        "repo": cast("str", repo),
        "commit": cast("str", commit),
        "rel_path": rel_path_value,
        "binding_id": binding_id_value,
        "scope_id": scope_id_value,
        "name": name_value,
        "binding_kind": cast("str", binding_kind),
    }
    return scope_key, binding_id_value, payload


def _build_binding_index(
    bindings: list[dict[str, object]],
) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[str, dict[str, object]]]:
    by_scope_name: dict[tuple[str, str, str], dict[str, object]] = {}
    by_id: dict[str, dict[str, object]] = {}
    for row in bindings:
        parsed = _binding_payload_from_row(row)
        if parsed is None:
            continue
        scope_key, binding_id_value, payload = parsed
        by_scope_name[scope_key] = payload
        by_id[binding_id_value] = payload
    return by_scope_name, by_id


def _build_resolution_map(
    resolutions: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_src: dict[str, dict[str, object]] = {}
    for row in resolutions:
        src_binding_id = _coerce_str(row.get("src_binding_id"))
        dst_binding_id = _coerce_str(row.get("dst_binding_id"))
        kind = _coerce_str(row.get("kind"))
        if src_binding_id is None or dst_binding_id is None or kind is None:
            continue
        confidence = _coerce_float(row.get("confidence"))
        reason = _coerce_str(row.get("reason"))
        existing = by_src.get(src_binding_id)
        existing_conf = _coerce_float(existing.get("confidence")) if existing else None
        prefer = existing is None or (
            confidence is not None and (existing_conf is None or confidence > existing_conf)
        )
        if prefer:
            by_src[src_binding_id] = {
                "dst_binding_id": dst_binding_id,
                "kind": kind,
                "confidence": confidence,
                "reason": reason,
            }
    return by_src


def _resolve_binding_for_event(
    *,
    rel_path: str,
    scope_id: str,
    name: str,
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    binding = bindings_by_scope.get((rel_path, scope_id, name))
    if binding is None:
        return None
    binding_id = _coerce_str(binding.get("binding_id"))
    binding_kind = _coerce_str(binding.get("binding_kind"))
    if binding_id is None or binding_kind is None:
        return None
    resolved_id = binding_id
    resolution = resolution_map.get(binding_id)
    if resolution is not None:
        resolved_id = _coerce_str(resolution.get("dst_binding_id")) or resolved_id
    return {
        "binding_id": binding_id,
        "binding_kind": binding_kind,
        "resolved_binding_id": resolved_id,
        "resolution": resolution,
    }


def _expected_scope_type(kind: str | None) -> str | None:
    if kind == "MODULE":
        return "MODULE"
    if kind == "CLASS":
        return "CLASS"
    if kind is None:
        return None
    return "FUNCTION"


def _scope_candidates(
    scopes: list[dict[str, object]],
    *,
    scope_type: str | None,
) -> list[dict[str, object]]:
    if scope_type is None:
        return scopes
    typed = [scope for scope in scopes if scope.get("scope_type") == scope_type]
    return typed if typed else scopes


def _span_length(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return max(end - start, 0)


def _span_contains(
    span_start: int | None,
    span_end: int | None,
    unit_start: int | None,
    unit_end: int | None,
) -> bool:
    if span_start is None or span_end is None or unit_start is None or unit_end is None:
        return False
    return span_start <= unit_start and span_end >= unit_end


def _select_scope_by_span(
    scopes: list[dict[str, object]],
    *,
    unit_start: int | None,
    unit_end: int | None,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for scope in scopes:
        scope_id = _coerce_str(scope.get("scope_id"))
        span_start = _coerce_int(scope.get("span_start_byte"))
        span_end = _coerce_int(scope.get("span_end_byte"))
        if scope_id is None or not _span_contains(span_start, span_end, unit_start, unit_end):
            continue
        span_len = _span_length(span_start, span_end)
        sort_key = span_len if span_len is not None else 2**63
        candidates.append((sort_key, scope_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _select_scope_by_lineno(
    scopes: list[dict[str, object]],
    *,
    lineno: int | None,
) -> str | None:
    if lineno is None:
        return None
    candidates: list[tuple[int, str]] = []
    for scope in scopes:
        scope_id = _coerce_str(scope.get("scope_id"))
        scope_line = _coerce_int(scope.get("lineno"))
        if scope_id is None or scope_line is None:
            continue
        candidates.append((abs(scope_line - lineno), scope_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _select_scope_for_unit(
    scopes: list[dict[str, object]],
    *,
    unit_kind: str | None,
    unit_lineno: int | None,
    unit_start: int | None,
    unit_end: int | None,
) -> str | None:
    target_type = _expected_scope_type(unit_kind)
    candidates = _scope_candidates(scopes, scope_type=target_type)
    scope_id = _select_scope_by_span(candidates, unit_start=unit_start, unit_end=unit_end)
    if scope_id is not None:
        return scope_id
    scope_id = _select_scope_by_lineno(candidates, lineno=unit_lineno)
    if scope_id is not None:
        return scope_id
    module_scopes = _scope_candidates(scopes, scope_type="MODULE")
    return _select_scope_by_lineno(module_scopes, lineno=unit_lineno)


def _scope_for_ast_event(
    scopes: list[dict[str, object]],
    event: Mapping[str, object],
) -> str | None:
    scope_id = _select_scope_by_span(
        scopes,
        unit_start=_coerce_int(event.get("start_byte")),
        unit_end=_coerce_int(event.get("end_byte")),
    )
    if scope_id is not None:
        return scope_id
    return _select_scope_by_lineno(scopes, lineno=_coerce_int(event.get("lineno")))


def _ast_binding_context_for_event(
    event: Mapping[str, object],
    *,
    scopes_by_path: Mapping[str, list[dict[str, object]]],
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    binding_meta: Mapping[str, Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> _AstBindingContext | None:
    rel_path = _coerce_str(event.get("rel_path"))
    scopes_for_path = scopes_by_path.get(rel_path) if rel_path else None
    scope_id = _scope_for_ast_event(scopes_for_path, event) if scopes_for_path else None
    name = _coerce_str(event.get("name")) if scope_id else None
    if _has_missing(rel_path, scope_id, name):
        return None
    rel_path_value = cast("str", rel_path)
    scope_id_value = cast("str", scope_id)
    name_value = cast("str", name)
    binding_info = _resolve_binding_for_event(
        rel_path=rel_path_value,
        scope_id=scope_id_value,
        name=name_value,
        bindings_by_scope=bindings_by_scope,
        resolution_map=resolution_map,
    )
    if binding_info is None:
        return None
    return _binding_context_from_info(
        rel_path=rel_path_value,
        binding_info=binding_info,
        binding_meta=binding_meta,
    )


def _binding_context_from_info(
    *,
    rel_path: str,
    binding_info: Mapping[str, object],
    binding_meta: Mapping[str, Mapping[str, object]],
) -> _AstBindingContext | None:
    binding_kind = _coerce_str(binding_info.get("binding_kind"))
    resolved_id = _coerce_str(binding_info.get("resolved_binding_id"))
    meta = binding_meta.get(resolved_id) if resolved_id else None
    repo = _coerce_str(meta.get("repo")) if meta else None
    commit = _coerce_str(meta.get("commit")) if meta else None
    if _has_missing(binding_kind, resolved_id, meta, repo, commit):
        return None
    resolution_payload = binding_info.get("resolution")
    resolution = resolution_payload if isinstance(resolution_payload, Mapping) else None
    return _AstBindingContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=rel_path,
        binding_meta=cast("Mapping[str, object]", meta),
        binding_kind=cast("str", binding_kind),
        resolution=resolution,
    )


def _ast_binding_edge_row(
    *,
    event: Mapping[str, object],
    context: _AstBindingContext,
) -> dict[str, object]:
    node_hash = _coerce_str(event.get("node_hash"))
    if node_hash is None:
        return {}
    src_cpg_node_id = cpg_node_id(AST_NODES_TABLE_KEY, {"hash": node_hash})
    dst_cpg_node_id = _binding_cpg_id(context.binding_meta)
    if dst_cpg_node_id is None:
        return {}
    event_kind = _coerce_str(event.get("event_kind")) or ""
    edge_kind = "BINDS_DEF" if event_kind == "DEF" else "BINDS_USE"
    extras = {
        "name": event.get("name"),
        "ctx": event.get("ctx"),
        "ast_node_type": event.get("node_type"),
        "binding_kind": context.binding_kind,
        "resolution_kind": _coerce_str(context.resolution.get("kind"))
        if context.resolution
        else None,
        "resolution_reason": _coerce_str(context.resolution.get("reason"))
        if context.resolution
        else None,
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_ast_binding",
        {
            "node_hash": node_hash,
            "binding_id": context.binding_meta.get("binding_id"),
            "edge_kind": edge_kind,
        },
    )
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": edge_kind,
        "edge_layer": "SYMBOL",
        "rel_path": context.rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _ast_event_kind(node_type: str, ctx: str | None) -> str | None:
    if node_type == "Name":
        if ctx == "store":
            return "DEF"
        if ctx == "load":
            return "USE"
        return None
    if node_type in {"arg", "FunctionDef", "AsyncFunctionDef", "ClassDef", "alias"}:
        return "DEF"
    return None


def _ast_binding_name(node_type: str, row: Mapping[str, object]) -> str | None:
    if node_type == "Name":
        return _coerce_str(row.get("identifier"))
    if node_type == "arg":
        return _coerce_str(row.get("name"))
    if node_type in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}:
        return _coerce_str(row.get("name"))
    if node_type == "alias":
        alias_name = _coerce_str(row.get("asname"))
        if alias_name:
            return alias_name
        imported = _coerce_str(row.get("imported"))
        if imported:
            return imported.rsplit(".", maxsplit=1)[-1]
    return None


def _ast_event_row(row: Mapping[str, object]) -> dict[str, object] | None:
    node_type = _coerce_str(row.get("node_type"))
    if node_type is None:
        return None
    ctx = _coerce_str(row.get("ctx"))
    event_kind = _ast_event_kind(node_type, ctx)
    if event_kind is None:
        return None
    name = _ast_binding_name(node_type, row)
    if name is None:
        return None
    node_hash = _coerce_str(row.get("hash"))
    rel_path = _coerce_str(row.get("path"))
    if node_hash is None or rel_path is None:
        return None
    return {
        "rel_path": rel_path,
        "node_hash": node_hash,
        "event_kind": event_kind,
        "name": name,
        "ctx": ctx,
        "node_type": node_type,
        "start_byte": _coerce_int(row.get("start_byte")),
        "end_byte": _coerce_int(row.get("end_byte")),
        "lineno": _coerce_int(row.get("lineno")),
    }


def _binding_cpg_id(binding_meta: Mapping[str, object]) -> int | None:
    repo = _coerce_str(binding_meta.get("repo"))
    commit = _coerce_str(binding_meta.get("commit"))
    rel_path = _coerce_str(binding_meta.get("rel_path"))
    binding_id = _coerce_str(binding_meta.get("binding_id"))
    if repo is None or commit is None or rel_path is None or binding_id is None:
        return None
    pk_values = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "binding_id": binding_id,
    }
    return cpg_node_id(PY_SYM_BINDINGS_TABLE_KEY, pk_values)


def _scopes_by_path(scope_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for scope in scope_rows:
        rel_path = _coerce_str(scope.get("rel_path"))
        if rel_path is None:
            continue
        by_path[rel_path].append(scope)
    return by_path


def _scope_qualname_from_qualpath(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    cleaned = value.replace("::", ".")
    cleaned = _QUALPATH_SUFFIX_RE.sub("", cleaned)
    return cleaned or None


def _record_diagnostics(
    diagnostics: dict[str, object] | None,
    key: str,
    *,
    expected_edges: int,
    produced_edges: int,
) -> None:
    if diagnostics is None:
        return
    dropped = max(expected_edges - produced_edges, 0)
    diagnostics[key] = OverlayEdgeDiagnostics(
        expected_edges=expected_edges,
        produced_edges=produced_edges,
        dropped_edges=dropped,
    )


def _collect_rows(frame: pa.Table, *, columns: Sequence[str]) -> list[dict[str, object]]:
    if not set(columns).issubset(frame.column_names):
        return []
    return [{column: row.get(column) for column in columns} for row in table_rows(frame)]


def _empty_edges() -> pa.Table:
    return empty_table_for_table(CPG_EDGES_TABLE_KEY)


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _has_missing(*values: object) -> bool:
    return any(value is None for value in values)


_QUALPATH_SUFFIX_RE = re.compile(r"#\d+")


@dataclass(frozen=True)
class _AstBindingContext:
    repo: str
    commit: str
    rel_path: str
    binding_meta: Mapping[str, object]
    binding_kind: str
    resolution: Mapping[str, object] | None


__all__ = [
    "OverlayEdgeDiagnostics",
    "cpg2_edges__ast_binding_edges",
    "cpg2_edges__py_sym_binding_edges",
    "cpg2_edges__py_sym_binding_symbol_edges",
    "cpg2_edges__py_sym_namespace_edges",
    "cpg2_edges__py_sym_resolution_edges",
    "cpg2_edges__py_sym_scope_edges",
]
