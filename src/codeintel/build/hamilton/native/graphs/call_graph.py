"""Call graph relation sources for graph targets."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"

_FUNCTION_KINDS: tuple[str, ...] = ("function", "method")


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


def _module_by_path(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        rel_path = row.get("path")
        module_name = row.get("module")
        language = row.get("language")
        if language not in {None, "python"}:
            continue
        if isinstance(rel_path, str) and isinstance(module_name, str):
            module_by_path[rel_path] = module_name
    return module_by_path


def _call_graph_indices(
    goids_frame: pl.DataFrame,
    module_by_path: dict[str, str],
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[int, str]]:
    goid_by_qualname: dict[str, int] = {}
    local_name_map: dict[str, dict[str, int]] = {}
    goid_language: dict[int, str] = {}
    for row in goids_frame.iter_rows(named=True):
        if row.get("kind") not in {"function", "method"}:
            continue
        qualname = row.get("qualname")
        rel_path = row.get("rel_path")
        if not isinstance(qualname, str) or not isinstance(rel_path, str):
            continue
        module_name = module_by_path.get(rel_path)
        if not module_name:
            continue
        goid = row.get("goid_h128")
        if not isinstance(goid, int):
            continue
        goid_by_qualname[qualname] = goid
        language = row.get("language")
        if isinstance(language, str):
            goid_language[goid] = language
        local_name = qualname.split(".")[-1]
        if qualname == f"{module_name}.{local_name}":
            local_map = local_name_map.setdefault(module_name, {})
            local_map.setdefault(local_name, goid)
    return goid_by_qualname, local_name_map, goid_language


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
                    "evidence_json": json.dumps({"callee_name": callee_name}),
                }
            )
    return edge_rows


def call_graph_nodes_compute(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> TabularFrame:
    """Build call graph nodes from core.goids and parsed ASTs.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed call graph nodes.
    """
    goids = tabular_to_lazyframe(q__core__goids).collect()
    if goids.is_empty():
        return empty_frame_for_table(CALL_GRAPH_NODES_TABLE_KEY)

    modules = tabular_to_lazyframe(q__core__modules).collect()
    module_by_path = _module_by_path(modules)

    function_map: dict[tuple[str, str], _FunctionDefInfo] = {}
    for rel_path, module_name in module_by_path.items():
        module_path = Path(env.snapshot.repo_root) / rel_path
        parsed = parse_python_module(module_path)
        if parsed is None:
            continue
        _, tree = parsed
        for info in _collect_function_defs(tree, module_name):
            function_map[rel_path, info.qualname] = info

    function_rows = [
        {
            "rel_path": rel_path,
            "qualname": qualname,
            "arity": info.arity,
            "is_public": info.is_public,
        }
        for (rel_path, qualname), info in function_map.items()
    ]
    frame = goids.filter(pl.col("kind").is_in(_FUNCTION_KINDS))
    qualname_public_expr = ~(pl.col("qualname").str.split(".").list.last().str.starts_with("_"))
    if function_rows:
        enrich = pl.DataFrame(function_rows)
        frame = frame.join(enrich, on=["rel_path", "qualname"], how="left")
        arity_expr = pl.coalesce([pl.col("arity"), pl.lit(0)]).cast(pl.Int64)
        is_public_expr = (
            pl.when(pl.col("is_public").is_null())
            .then(qualname_public_expr)
            .otherwise(pl.col("is_public"))
        )
    else:
        arity_expr = pl.lit(0).cast(pl.Int64)
        is_public_expr = qualname_public_expr

    frame = frame.with_columns(
        arity_expr.alias("arity"),
        is_public_expr.alias("is_public"),
    )
    return frame.select(
        [
            "goid_h128",
            "language",
            "kind",
            "arity",
            "is_public",
            "rel_path",
        ]
    ).lazy()


def call_graph_edges_compute(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> TabularFrame:
    """Build a minimal call graph edges frame from local name resolution.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed call graph edges.
    """
    goids = tabular_to_lazyframe(q__core__goids).collect()
    if goids.is_empty():
        return empty_frame_for_table(CALL_GRAPH_EDGES_TABLE_KEY)

    modules = tabular_to_lazyframe(q__core__modules).collect()
    module_by_path = _module_by_path(modules)
    goid_by_qualname, local_name_map, goid_language = _call_graph_indices(
        goids,
        module_by_path,
    )
    edge_context = _CallGraphEdgeContext(
        env=env,
        goid_by_qualname=goid_by_qualname,
        local_name_map=local_name_map,
        goid_language=goid_language,
    )

    edge_rows: list[dict[str, object]] = []
    for rel_path, module_name in module_by_path.items():
        edge_rows.extend(
            _edge_rows_for_module(
                edge_context,
                rel_path=rel_path,
                module_name=module_name,
            )
        )

    if not edge_rows:
        return empty_frame_for_table(CALL_GRAPH_EDGES_TABLE_KEY)
    frame = pl.DataFrame(edge_rows)
    return frame.lazy().select(
        [
            "repo",
            "commit",
            "caller_goid_h128",
            "callee_goid_h128",
            "callsite_path",
            "callsite_line",
            "callsite_col",
            "language",
            "kind",
            "resolved_via",
            "confidence",
            "evidence_json",
        ]
    )


def call_graph_nodes_existing(env: BuildEnv) -> TabularFrame:
    """Load call graph nodes from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing call graph nodes.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=CALL_GRAPH_NODES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def call_graph_edges_existing(env: BuildEnv) -> TabularFrame:
    """Load call graph edges from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing call graph edges.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=CALL_GRAPH_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def call_graph_nodes_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for call graph nodes.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for call graph nodes.
    """
    _ = env
    return empty_frame_for_table(CALL_GRAPH_NODES_TABLE_KEY)


def call_graph_edges_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for call graph edges.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for call graph edges.
    """
    _ = env
    return empty_frame_for_table(CALL_GRAPH_EDGES_TABLE_KEY)


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
