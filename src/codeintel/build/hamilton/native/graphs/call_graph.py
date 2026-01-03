"""Call graph relation sources for graph targets."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.columnar.rows import empty_reader_for_table, record_batch_reader_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
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
    if modules_frame.is_empty():
        return module_by_path
    required = {"path", "module"}
    if not required.issubset(set(modules_frame.columns)):
        return module_by_path
    filtered = modules_frame
    if "language" in filtered.columns:
        filtered = filtered.filter(pl.col("language").is_null() | (pl.col("language") == "python"))
    filtered = filtered.select(["path", "module"]).with_columns(
        pl.col("path").cast(pl.Utf8, strict=False),
        pl.col("module").cast(pl.Utf8, strict=False),
    )
    filtered = filtered.filter(
        pl.col("path").is_not_null()
        & pl.col("module").is_not_null()
        & (pl.col("path").str.len_chars() > 0)
        & (pl.col("module").str.len_chars() > 0)
    )
    if filtered.is_empty():
        return module_by_path
    data = filtered.to_dict(as_series=False)
    for rel_path, module_name in zip(data["path"], data["module"], strict=True):
        module_by_path[str(rel_path)] = str(module_name)
    return module_by_path


def _call_graph_indices(
    goids_frame: pl.DataFrame,
    module_by_path: dict[str, str],
) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[int, str]]:
    goid_by_qualname: dict[str, int] = {}
    local_name_map: dict[str, dict[str, int]] = {}
    goid_language: dict[int, str] = {}
    data = _call_graph_index_rows(goids_frame, module_by_path)
    if data is None:
        return goid_by_qualname, local_name_map, goid_language
    languages = data.get("language")
    for idx, (qualname, rel_path, goid_raw) in enumerate(
        zip(data["qualname"], data["rel_path"], data["goid_h128"], strict=True)
    ):
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
        if languages is not None:
            language_raw = languages[idx]
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
    goids_frame: pl.DataFrame,
    module_by_path: dict[str, str],
) -> dict[str, list[object]] | None:
    if goids_frame.is_empty() or not module_by_path:
        return None
    required = {"kind", "qualname", "rel_path", "goid_h128"}
    if not required.issubset(set(goids_frame.columns)):
        return None
    filtered = goids_frame.filter(pl.col("kind").is_in(_FUNCTION_KINDS))
    if "rel_path" in filtered.columns:
        filtered = filtered.filter(pl.col("rel_path").is_in(list(module_by_path)))
    if filtered.is_empty():
        return None
    columns = ["qualname", "rel_path", "goid_h128"]
    if "language" in filtered.columns:
        columns.append("language")
    return filtered.select(columns).to_dict(as_series=False)


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
    goids = (
        tabular_to_lazyframe(q__core__goids)
        .select(["goid_h128", "language", "kind", "rel_path", "qualname"])
        .collect()
    )
    if goids.is_empty():
        return empty_frame_for_table(CALL_GRAPH_NODES_TABLE_KEY)

    modules = (
        tabular_to_lazyframe(q__core__modules).select(["path", "module", "language"]).collect()
    )
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
) -> InferableTabularInput:
    """Build a minimal call graph edges frame from local name resolution.

    Returns
    -------
    InferableTabularInput
        Tabular input for computed call graph edges.
    """
    goids = (
        tabular_to_lazyframe(q__core__goids)
        .select(["goid_h128", "language", "kind", "rel_path", "qualname"])
        .collect()
    )
    if goids.is_empty():
        return empty_reader_for_table(CALL_GRAPH_EDGES_TABLE_KEY)

    modules = (
        tabular_to_lazyframe(q__core__modules).select(["path", "module", "language"]).collect()
    )
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

    reader, _ = record_batch_reader_for_rows(
        CALL_GRAPH_EDGES_TABLE_KEY,
        _edge_rows(edge_context, module_by_path=module_by_path),
    )
    return reader


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
    return empty_reader_for_table(CALL_GRAPH_NODES_TABLE_KEY)


def call_graph_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for call graph edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for call graph edges.
    """
    _ = env
    return empty_reader_for_table(CALL_GRAPH_EDGES_TABLE_KEY)


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
