"""CFG/DFG relation sources for graph targets."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyarrow as pa

from codeintel.build.graphs.compute.cfg import build_cfg, cfg_to_rows
from codeintel.build.graphs.compute.dfg import build_dfg, dfg_to_rows
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.compute_filters import (
    filter_function_ast_nodes,
    filter_python_goids,
)
from codeintel.build.hamilton.native.graphs.filter_helpers import plan_filter_or_fallback
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import iter_array_values, iter_rows
from codeintel.build.tabular.compute_helpers import cast_array
from codeintel.build.tabular.compute_masks import non_empty_string_expr, non_empty_string_mask
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.data_models.rows import CFGBlockRow, CFGEdgeRow, DFGEdgeRow
from codeintel.core.spans import normalize_line_span
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef

CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"
_FUNCTION_GOID_TYPE = pa.decimal128(38, 0)
_ASCENDING: Literal["ascending"] = "ascending"


@dataclass(frozen=True, slots=True)
class _FunctionNodeInfo:
    node: ast.FunctionDef | ast.AsyncFunctionDef
    start_line: int
    end_line: int | None
    name: str


@dataclass(frozen=True, slots=True)
class _FunctionGoidInfo:
    goid: int
    rel_path: str
    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class _CfgDfgAnalysis:
    cfg_blocks: tuple[CFGBlockRow, ...]
    cfg_edges: tuple[CFGEdgeRow, ...]
    dfg_edges: tuple[DFGEdgeRow, ...]


def _filter_non_empty_paths(table: pa.Table) -> pa.Table:
    if table.num_rows == 0 or "path" not in table.column_names:
        return table

    def _mask(value_table: pa.Table) -> pa.Array | pa.ChunkedArray:
        return non_empty_string_mask(value_table.column("path"))

    return plan_filter_or_fallback(
        table,
        non_empty_string_expr("path"),
        fallback_mask=_mask,
    )


def _collect_ast_function_keys(
    ast_nodes_table: pa.Table,
) -> tuple[dict[str, set[tuple[int, str]]], set[str]]:
    function_keys_by_path: dict[str, set[tuple[int, str]]] = {}
    paths: set[str] = set()
    required = {"path", "node_type", "name", "lineno"}
    if ast_nodes_table.num_rows == 0 or not required.issubset(set(ast_nodes_table.column_names)):
        return function_keys_by_path, paths
    path_table = _filter_non_empty_paths(ast_nodes_table)
    for path in iter_array_values(path_table.column("path")):
        if isinstance(path, str) and path:
            paths.add(path)
    filtered = filter_function_ast_nodes(ast_nodes_table)
    for row in iter_rows(filtered):
        path = row.get("path")
        name = row.get("name")
        lineno = row.get("lineno")
        if not isinstance(path, str) or not path:
            continue
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(lineno, int):
            continue
        function_keys_by_path.setdefault(path, set()).add((lineno, name))
    return function_keys_by_path, paths


def _collect_goids_by_path(
    goids_table: pa.Table,
    function_keys_by_path: dict[str, set[tuple[int, str]]],
) -> dict[str, list[_FunctionGoidInfo]]:
    goids_by_path: dict[str, list[_FunctionGoidInfo]] = {}
    if goids_table.num_rows == 0:
        return goids_by_path
    required = {"kind", "rel_path", "qualname", "goid_h128", "start_line"}
    if not required.issubset(set(goids_table.column_names)):
        return goids_by_path
    filtered = filter_python_goids(goids_table)
    for row in iter_rows(filtered):
        if row.get("kind") not in {"function", "method"}:
            continue
        language = row.get("language")
        if language not in {None, "python"}:
            continue
        rel_path = row.get("rel_path")
        qualname = row.get("qualname")
        goid_raw = row.get("goid_h128")
        start_line = row.get("start_line")
        end_line = row.get("end_line")
        if not isinstance(rel_path, str) or not isinstance(qualname, str):
            continue
        if not isinstance(start_line, int):
            continue
        name = qualname.split(".")[-1]
        key_set = function_keys_by_path.get(rel_path)
        if key_set is not None and (start_line, name) not in key_set:
            continue
        goid_value = normalize_decimal_id(goid_raw)
        if goid_value is None:
            continue
        _, resolved_end = normalize_line_span(
            start_line,
            end_line if isinstance(end_line, int) else None,
        )
        info = _FunctionGoidInfo(
            goid=int(goid_value),
            rel_path=rel_path,
            name=name,
            start_line=start_line,
            end_line=resolved_end,
        )
        goids_by_path.setdefault(rel_path, []).append(info)
    return goids_by_path


def _build_cfg_dfg_rows(
    snapshot: SnapshotRef,
    repo_root: Path,
    goids_by_path: dict[str, list[_FunctionGoidInfo]],
    paths: set[str],
) -> tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]:
    cfg_blocks: list[CFGBlockRow] = []
    cfg_edges: list[CFGEdgeRow] = []
    dfg_edges: list[DFGEdgeRow] = []
    for rel_path, goid_entries in goids_by_path.items():
        if rel_path not in paths:
            continue
        module_path = repo_root / rel_path
        parsed = parse_python_module(module_path)
        if parsed is None:
            continue
        _, tree = parsed
        nodes_by_line = _collect_function_nodes(tree)
        for info in goid_entries:
            node_info = nodes_by_line.get((info.start_line, info.name))
            if node_info is None:
                continue
            cfg_result = build_cfg(info.goid, node_info.node, info.rel_path)
            block_rows, edge_rows = cfg_to_rows(
                cfg_result,
                snapshot,
                info.rel_path,
                info.start_line,
                info.end_line,
            )
            cfg_blocks.extend(block_rows)
            cfg_edges.extend(edge_rows)
            dfg_result = build_dfg(info.goid, cfg_result.blocks, cfg_result.edges)
            dfg_edges.extend(dfg_to_rows(dfg_result, snapshot.repo, snapshot.commit))
    return cfg_blocks, cfg_edges, dfg_edges


def _collect_function_nodes(tree: ast.AST) -> dict[tuple[int, str], _FunctionNodeInfo]:
    nodes: dict[tuple[int, str], _FunctionNodeInfo] = {}

    def _visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start_line = getattr(node, "lineno", None)
            if isinstance(start_line, int):
                start_line = max(start_line - 1, 0)
                end_line = getattr(node, "end_lineno", None)
                normalized_end = max(end_line - 1, 0) if isinstance(end_line, int) else None
                nodes[start_line, node.name] = _FunctionNodeInfo(
                    node=node,
                    start_line=start_line,
                    end_line=normalized_end,
                    name=node.name,
                )
            for child in node.body:
                _visit(child)
            return
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                _visit(child)
            return
        if isinstance(node, ast.Module):
            for child in node.body:
                _visit(child)

    _visit(tree)
    return nodes


def cfg_dfg_analysis(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__ast_nodes: InferableTabularInput,
) -> _CfgDfgAnalysis:
    """Build CFG/DFG row payloads from AST and goid inputs.

    Parameters
    ----------
    env
        Build environment with repository snapshot info.
    q__core__goids
        Core goid table input.
    q__core__ast_nodes
        Core AST nodes table input.

    Returns
    -------
    _CfgDfgAnalysis
        Container of CFG blocks/edges and DFG edges rows.
    """
    goids_table = tabular_to_scoped_table(
        q__core__goids,
        columns=[
            "goid_h128",
            "rel_path",
            "qualname",
            "start_line",
            "end_line",
            "kind",
            "language",
        ],
        scope=None,
        require_scope_columns=False,
    )
    goids_table = filter_python_goids(goids_table)
    if goids_table.num_rows == 0:
        return _CfgDfgAnalysis(cfg_blocks=(), cfg_edges=(), dfg_edges=())

    ast_nodes_table = tabular_to_scoped_table(
        q__core__ast_nodes,
        columns=["path", "node_type", "name", "lineno"],
        scope=None,
        require_scope_columns=False,
    )
    function_keys_by_path, paths = _collect_ast_function_keys(ast_nodes_table)
    goids_by_path = _collect_goids_by_path(goids_table, function_keys_by_path)
    resolved_paths = paths or set(goids_by_path)
    repo_root = Path(env.snapshot.repo_root)
    cfg_blocks, cfg_edges, dfg_edges = _build_cfg_dfg_rows(
        env.snapshot,
        repo_root,
        goids_by_path,
        resolved_paths,
    )

    return _CfgDfgAnalysis(
        cfg_blocks=tuple(cfg_blocks),
        cfg_edges=tuple(cfg_edges),
        dfg_edges=tuple(dfg_edges),
    )


def cfg_blocks_compute(cfg_dfg_analysis: _CfgDfgAnalysis) -> InferableTabularInput:
    """Build CFG blocks from parsed AST inputs.

    Returns
    -------
    InferableTabularInput
        Arrow reader of CFG block rows.
    """
    if not cfg_dfg_analysis.cfg_blocks:
        return empty_table_for_table(CFG_BLOCKS_TABLE_KEY)
    rows = (dataclasses.asdict(row) for row in cfg_dfg_analysis.cfg_blocks)
    table, _ = table_for_rows(CFG_BLOCKS_TABLE_KEY, rows)
    table = _cast_function_goid(table)
    result = finalize_table(
        table,
        spec=FinalizeSpec(
            table_key=CFG_BLOCKS_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(CFG_BLOCKS_TABLE_KEY),
            order_by=_order_by_for_table(CFG_BLOCKS_TABLE_KEY),
        ),
    )
    return result.good


def cfg_edges_compute(cfg_dfg_analysis: _CfgDfgAnalysis) -> InferableTabularInput:
    """Build CFG edges from parsed AST inputs.

    Returns
    -------
    InferableTabularInput
        Tabular input of CFG edge rows.
    """
    if not cfg_dfg_analysis.cfg_edges:
        return empty_table_for_table(CFG_EDGES_TABLE_KEY)
    rows = (dataclasses.asdict(row) for row in cfg_dfg_analysis.cfg_edges)
    table, _ = table_for_rows(CFG_EDGES_TABLE_KEY, rows)
    table = _cast_function_goid(table)
    result = finalize_table(
        table,
        spec=FinalizeSpec(
            table_key=CFG_EDGES_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(CFG_EDGES_TABLE_KEY),
            order_by=_order_by_for_table(CFG_EDGES_TABLE_KEY),
        ),
    )
    return result.good


def dfg_edges_compute(cfg_dfg_analysis: _CfgDfgAnalysis) -> InferableTabularInput:
    """Build DFG edges from parsed AST inputs.

    Returns
    -------
    InferableTabularInput
        Tabular input of DFG edge rows.
    """
    if not cfg_dfg_analysis.dfg_edges:
        return empty_table_for_table(DFG_EDGES_TABLE_KEY)
    rows = (dataclasses.asdict(row) for row in cfg_dfg_analysis.dfg_edges)
    table, _ = table_for_rows(DFG_EDGES_TABLE_KEY, rows)
    table = _cast_function_goid(table)
    result = finalize_table(
        table,
        spec=FinalizeSpec(
            table_key=DFG_EDGES_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(DFG_EDGES_TABLE_KEY),
            order_by=_order_by_for_table(DFG_EDGES_TABLE_KEY),
        ),
    )
    return result.good


def cfg_blocks_existing(env: BuildEnv) -> InferableTabularInput:
    """Load CFG blocks from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing CFG blocks.
    """
    table = load_snapshot_tabular(
        env=env,
        table_key=CFG_BLOCKS_TABLE_KEY,
        snapshot_id=env.commit,
    )
    return _cast_function_goid(table)


def cfg_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load CFG edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing CFG edges.
    """
    table = load_snapshot_tabular(
        env=env,
        table_key=CFG_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )
    return _cast_function_goid(table)


def dfg_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load DFG edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing DFG edges.
    """
    table = load_snapshot_tabular(
        env=env,
        table_key=DFG_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )
    return _cast_function_goid(table)


def cfg_blocks_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for CFG blocks.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for CFG blocks.
    """
    _ = env
    return empty_table_for_table(CFG_BLOCKS_TABLE_KEY)


def cfg_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for CFG edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for CFG edges.
    """
    _ = env
    return empty_table_for_table(CFG_EDGES_TABLE_KEY)


def dfg_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for DFG edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for DFG edges.
    """
    _ = env
    return empty_table_for_table(DFG_EDGES_TABLE_KEY)


def _cast_function_goid(table: pa.Table) -> pa.Table:
    if "function_goid_h128" not in table.column_names:
        return table
    index = table.schema.get_field_index("function_goid_h128")
    if index == -1:
        return table
    field = table.schema.field(index)
    if field.type == _FUNCTION_GOID_TYPE:
        return table
    column = table.column(index)
    casted = cast_array(column, _FUNCTION_GOID_TYPE, safe=True)
    return table.set_column(index, field.name, casted)


def _key_fields_for_table(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except (KeyError, RuntimeError, TypeError):
        return ()
    if schema is None or not schema.primary_key:
        return ()
    return tuple(schema.primary_key)


def _order_by_for_table(table_key: str) -> tuple[SortKey, ...]:
    key_fields = _key_fields_for_table(table_key)
    if not key_fields:
        return ()
    return tuple((field, _ASCENDING) for field in key_fields)


__all__ = [
    "CFG_BLOCKS_TABLE_KEY",
    "CFG_EDGES_TABLE_KEY",
    "DFG_EDGES_TABLE_KEY",
    "cfg_blocks_compute",
    "cfg_blocks_empty",
    "cfg_blocks_existing",
    "cfg_dfg_analysis",
    "cfg_edges_compute",
    "cfg_edges_empty",
    "cfg_edges_existing",
    "dfg_edges_compute",
    "dfg_edges_empty",
    "dfg_edges_existing",
]
