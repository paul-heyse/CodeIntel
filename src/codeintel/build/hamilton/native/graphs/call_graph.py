"""Call graph relation sources for graph targets."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.compute_filters import (
    filter_python_goids,
    filter_python_modules,
)
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"

_FUNCTION_KINDS: tuple[str, ...] = ("function", "method")

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FunctionDefInfo:
    qualname: str
    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    arity: int
    is_public: bool
    start_line: int | None


@dataclass(frozen=True, slots=True)
class _CallGraphEdgeContext:
    env: BuildEnv
    goid_by_qualname: dict[str, int]
    local_name_map: dict[str, dict[str, int]]
    goid_language: dict[int, str]


def _function_arity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    total = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
    if args.vararg is not None:
        total += 1
    if args.kwarg is not None:
        total += 1
    return total


def _collect_function_defs(tree: ast.AST, module_name: str) -> list[_FunctionDefInfo]:
    results: list[_FunctionDefInfo] = []
    scope: list[str] = []

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            scope.append(node.name)
            for child in node.body:
                _visit(child)
            scope.pop()
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join([module_name, *scope, node.name]) if module_name else node.name
            results.append(
                _FunctionDefInfo(
                    qualname=qualname,
                    name=node.name,
                    node=node,
                    arity=_function_arity(node),
                    is_public=not node.name.startswith("_"),
                    start_line=getattr(node, "lineno", None),
                )
            )
            scope.append(node.name)
            for child in node.body:
                _visit(child)
            scope.pop()
            return
        if isinstance(node, ast.Module):
            for child in node.body:
                _visit(child)

    _visit(tree)
    return results


class _CallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def _visit_body(self, body: list[ast.stmt]) -> None:
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_body(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_body(node.body)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _module_by_path(modules_table: pa.Table) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    if modules_table.num_rows == 0:
        return module_by_path
    required = {"path", "module"}
    if not required.issubset(set(modules_table.column_names)):
        return module_by_path
    filtered = filter_python_modules(modules_table)
    for row in iter_rows(filtered):
        rel_path = row.get("path")
        module_name = row.get("module")
        language = row.get("language")
        if language not in {None, "python"}:
            continue
        if not isinstance(rel_path, str) or not rel_path:
            continue
        if not isinstance(module_name, str) or not module_name:
            continue
        module_by_path[rel_path] = module_name
    return module_by_path


def _call_graph_indices(
    goids_table: pa.Table,
    module_by_path: dict[str, str],
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[int, str]]:
    goid_by_qualname: dict[str, int] = {}
    local_name_map: dict[str, dict[str, int]] = {}
    goid_language: dict[int, str] = {}
    data = _call_graph_index_rows(goids_table, module_by_path)
    if data is None:
        return goid_by_qualname, local_name_map, goid_language
    for item in data:
        qualname = item.get("qualname")
        rel_path = item.get("rel_path")
        goid_raw = item.get("goid_h128")
        if qualname is None or rel_path is None:
            continue
        module_name = module_by_path.get(str(rel_path))
        if not module_name:
            continue
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        qualname_value = str(qualname)
        goid_by_qualname[qualname_value] = goid
        language_raw = item.get("language")
        if isinstance(language_raw, str):
            goid_language[goid] = language_raw
        _update_local_name_map(
            local_name_map,
            module_name=module_name,
            qualname=qualname_value,
            goid=goid,
        )
    return goid_by_qualname, local_name_map, goid_language


def _call_graph_index_rows(
    goids_table: pa.Table,
    module_by_path: dict[str, str],
) -> list[dict[str, object]] | None:
    if goids_table.num_rows == 0 or not module_by_path:
        return None
    required = {"kind", "qualname", "rel_path", "goid_h128"}
    if not required.issubset(set(goids_table.column_names)):
        return None
    filtered = filter_python_goids(goids_table)
    rows: list[dict[str, object]] = []
    for row in iter_rows(filtered):
        if row.get("kind") not in _FUNCTION_KINDS:
            continue
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str) or rel_path not in module_by_path:
            continue
        rows.append(row)
    return rows or None


def _update_local_name_map(
    local_name_map: dict[str, dict[str, int]],
    *,
    module_name: str,
    qualname: str,
    goid: int,
) -> None:
    local_name = qualname.rsplit(".", maxsplit=1)[-1]
    if qualname != f"{module_name}.{local_name}":
        return
    local_map = local_name_map.setdefault(module_name, {})
    local_map.setdefault(local_name, goid)


def _edge_rows_for_module(
    context: _CallGraphEdgeContext,
    *,
    rel_path: str,
    module_name: str,
) -> list[dict[str, object]]:
    module_path = Path(context.env.snapshot.repo_root) / rel_path
    parsed = parse_python_module(module_path)
    if parsed is None:
        return []
    _, tree = parsed
    edge_rows: list[dict[str, object]] = []
    for info in _collect_function_defs(tree, module_name):
        caller_goid = context.goid_by_qualname.get(info.qualname)
        if caller_goid is None:
            continue
        collector = _CallCollector()
        collector.visit(info.node)
        for call in collector.calls:
            callee_name = _call_name(call.func)
            if callee_name is None:
                continue
            callee_goid = context.local_name_map.get(module_name, {}).get(callee_name)
            if callee_goid is None:
                continue
            callsite_line = getattr(call, "lineno", 0) or 0
            callsite_col = getattr(call, "col_offset", 0) or 0
            edge_rows.append(
                {
                    "repo": context.env.repo,
                    "commit": context.env.commit,
                    "caller_goid_h128": caller_goid,
                    "callee_goid_h128": callee_goid,
                    "callsite_path": rel_path,
                    "callsite_line": int(callsite_line),
                    "callsite_col": int(callsite_col),
                    "language": context.goid_language.get(caller_goid, "python"),
                    "kind": "call",
                    "resolved_via": "local_name",
                    "confidence": 0.6,
                    "evidence_json": {"callee_name": callee_name},
                }
            )
    return edge_rows


def _edge_rows(
    context: _CallGraphEdgeContext,
    *,
    module_by_path: dict[str, str],
) -> Iterable[dict[str, object]]:
    for rel_path, module_name in module_by_path.items():
        yield from _edge_rows_for_module(
            context,
            rel_path=rel_path,
            module_name=module_name,
        )


def _function_definitions(
    env: BuildEnv,
    module_by_path: Mapping[str, str],
) -> dict[tuple[str, str], _FunctionDefInfo]:
    function_map: dict[tuple[str, str], _FunctionDefInfo] = {}
    for rel_path, module_name in module_by_path.items():
        module_path = Path(env.snapshot.repo_root) / rel_path
        parsed = parse_python_module(module_path)
        if parsed is None:
            continue
        _, tree = parsed
        for info in _collect_function_defs(tree, module_name):
            function_map[rel_path, info.qualname] = info
    return function_map


def _call_graph_node_rows(
    *,
    env: BuildEnv,
    goids_table: pa.Table,
    module_by_path: Mapping[str, str],
) -> list[dict[str, object]]:
    function_map = _function_definitions(env, module_by_path)
    function_rows = [
        {
            "rel_path": rel_path,
            "qualname": qualname,
            "arity": info.arity,
            "is_public": info.is_public,
        }
        for (rel_path, qualname), info in function_map.items()
    ]
    function_index = {
        (row["rel_path"], row["qualname"]): row for row in function_rows if "rel_path" in row
    }
    output_rows: list[dict[str, object]] = []
    matched_defs = 0
    total_defs = 0
    for row in iter_rows(goids_table):
        kind = row.get("kind")
        if kind not in _FUNCTION_KINDS:
            continue
        total_defs += 1
        rel_path = row.get("rel_path")
        qualname = row.get("qualname")
        if not isinstance(rel_path, str) or not isinstance(qualname, str):
            continue
        enrich = function_index.get((rel_path, qualname))
        if enrich is not None:
            matched_defs += 1
        arity = enrich.get("arity") if enrich is not None else 0
        is_public = enrich.get("is_public") if enrich is not None else None
        if is_public is None:
            last = qualname.rsplit(".", maxsplit=1)[-1]
            is_public = not last.startswith("_")
        output_rows.append(
            {
                "goid_h128": row.get("goid_h128"),
                "language": row.get("language"),
                "kind": kind,
                "arity": int(arity) if isinstance(arity, int) else 0,
                "is_public": bool(is_public),
                "rel_path": rel_path,
            }
        )
    if total_defs:
        LOG.info(
            "call_graph_nodes join coverage goids_to_defs matched=%d total=%d",
            matched_defs,
            total_defs,
        )
    return output_rows


def call_graph_nodes_compute(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> InferableTabularInput:
    """Build call graph nodes from core.goids and parsed ASTs.

    Returns
    -------
    InferableTabularInput
        Arrow reader for computed call graph nodes.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    goids_table = tabular_to_scoped_table(
        q__core__goids,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    if goids_table.num_rows == 0:
        return empty_table_for_table(CALL_GRAPH_NODES_TABLE_KEY)

    modules_table = tabular_to_scoped_table(
        q__core__modules,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    module_by_path = _module_by_path(modules_table)
    output_rows = _call_graph_node_rows(
        env=env,
        goids_table=goids_table,
        module_by_path=module_by_path,
    )
    table, _ = table_for_rows(CALL_GRAPH_NODES_TABLE_KEY, output_rows)
    result = finalize_table(
        table,
        spec=FinalizeSpec(table_key=CALL_GRAPH_NODES_TABLE_KEY, mode="strict"),
    )
    return result.good


def call_graph_edges_compute(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> InferableTabularInput:
    """Build a minimal call graph edges frame from local name resolution.

    Returns
    -------
    InferableTabularInput
        Tabular input for computed call graph edges.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    goids_table = tabular_to_scoped_table(
        q__core__goids,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    if goids_table.num_rows == 0:
        return empty_table_for_table(CALL_GRAPH_EDGES_TABLE_KEY)

    modules_table = tabular_to_scoped_table(
        q__core__modules,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    module_by_path = _module_by_path(modules_table)
    goid_by_qualname, local_name_map, goid_language = _call_graph_indices(
        goids_table,
        module_by_path,
    )
    edge_context = _CallGraphEdgeContext(
        env=env,
        goid_by_qualname=goid_by_qualname,
        local_name_map=local_name_map,
        goid_language=goid_language,
    )

    table, _ = table_for_rows(
        CALL_GRAPH_EDGES_TABLE_KEY,
        _edge_rows(edge_context, module_by_path=module_by_path),
    )
    result = finalize_table(
        table,
        spec=FinalizeSpec(table_key=CALL_GRAPH_EDGES_TABLE_KEY, mode="strict"),
    )
    return result.good


def call_graph_nodes_existing(env: BuildEnv) -> InferableTabularInput:
    """Load call graph nodes from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing call graph nodes.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=CALL_GRAPH_NODES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def call_graph_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load call graph edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing call graph edges.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=CALL_GRAPH_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def call_graph_nodes_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for call graph nodes.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for call graph nodes.
    """
    _ = env
    return empty_table_for_table(CALL_GRAPH_NODES_TABLE_KEY)


def call_graph_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for call graph edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for call graph edges.
    """
    _ = env
    return empty_table_for_table(CALL_GRAPH_EDGES_TABLE_KEY)


__all__ = [
    "CALL_GRAPH_EDGES_TABLE_KEY",
    "CALL_GRAPH_NODES_TABLE_KEY",
    "call_graph_edges_compute",
    "call_graph_edges_empty",
    "call_graph_edges_existing",
    "call_graph_nodes_compute",
    "call_graph_nodes_empty",
    "call_graph_nodes_existing",
]
