"""CFG/DFG relation sources for graph targets."""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from codeintel.build.graphs.compute.cfg import build_cfg, cfg_to_rows
from codeintel.build.graphs.compute.dfg import build_dfg, dfg_to_rows
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.columnar.rows import empty_reader_for_table, record_batch_reader_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.data_models.rows import CFGBlockRow, CFGEdgeRow, DFGEdgeRow
from codeintel.core.spans import normalize_line_span
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"


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


def _collect_ast_function_keys(
    ast_nodes_frame: pl.DataFrame,
) -> tuple[dict[str, set[tuple[int, str]]], set[str]]:
    function_keys_by_path: dict[str, set[tuple[int, str]]] = {}
    paths: set[str] = set()
    if ast_nodes_frame.is_empty() or "path" not in ast_nodes_frame.columns:
        return function_keys_by_path, paths
    path_values = ast_nodes_frame.get_column("path").drop_nulls().to_list()
    paths = {str(path) for path in path_values if isinstance(path, str) and path}
    if not {"node_type", "name", "lineno"}.issubset(set(ast_nodes_frame.columns)):
        return function_keys_by_path, paths
    functions = ast_nodes_frame.select(["path", "node_type", "name", "lineno"]).filter(
        pl.col("node_type").is_in(["FunctionDef", "AsyncFunctionDef"])
    )
    if functions.is_empty():
        return function_keys_by_path, paths
    data = functions.to_dict(as_series=False)
    for path, name, lineno in zip(
        data["path"],
        data["name"],
        data["lineno"],
        strict=True,
    ):
        if not isinstance(path, str) or not path:
            continue
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(lineno, int):
            continue
        function_keys_by_path.setdefault(path, set()).add((lineno, name))
    return function_keys_by_path, paths


def _collect_goids_by_path(
    goids_frame: pl.DataFrame,
    function_keys_by_path: dict[str, set[tuple[int, str]]],
) -> dict[str, list[_FunctionGoidInfo]]:
    goids_by_path: dict[str, list[_FunctionGoidInfo]] = {}
    if goids_frame.is_empty():
        return goids_by_path
    required = {"kind", "rel_path", "qualname", "goid_h128", "start_line"}
    if not required.issubset(set(goids_frame.columns)):
        return goids_by_path
    filtered = goids_frame.filter(pl.col("kind").is_in(["function", "method"]))
    if "language" in filtered.columns:
        filtered = filtered.filter(pl.col("language").is_null() | (pl.col("language") == "python"))
    if filtered.is_empty():
        return goids_by_path
    columns = ["rel_path", "qualname", "goid_h128", "start_line", "end_line"]
    data = filtered.select(columns).to_dict(as_series=False)
    for rel_path, qualname, goid_raw, start_line, end_line in zip(
        data["rel_path"],
        data["qualname"],
        data["goid_h128"],
        data["start_line"],
        data["end_line"],
        strict=True,
    ):
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
                info.rel_path,
                info.start_line,
                info.end_line,
            )
            cfg_blocks.extend(block_rows)
            cfg_edges.extend(edge_rows)
            dfg_result = build_dfg(info.goid, cfg_result.blocks, cfg_result.edges)
            dfg_edges.extend(dfg_to_rows(dfg_result))
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
    goids_frame = (
        tabular_to_lazyframe(q__core__goids)
        .select(
            [
                "goid_h128",
                "rel_path",
                "qualname",
                "start_line",
                "end_line",
                "kind",
                "language",
            ]
        )
        .collect()
    )
    if goids_frame.is_empty():
        return _CfgDfgAnalysis(cfg_blocks=(), cfg_edges=(), dfg_edges=())

    ast_nodes_frame = (
        tabular_to_lazyframe(q__core__ast_nodes)
        .select(["path", "node_type", "name", "lineno"])
        .collect()
    )
    function_keys_by_path, paths = _collect_ast_function_keys(ast_nodes_frame)
    goids_by_path = _collect_goids_by_path(goids_frame, function_keys_by_path)
    resolved_paths = paths or set(goids_by_path)
    repo_root = Path(env.snapshot.repo_root)
    cfg_blocks, cfg_edges, dfg_edges = _build_cfg_dfg_rows(
        repo_root,
        goids_by_path,
        resolved_paths,
    )

    return _CfgDfgAnalysis(
        cfg_blocks=tuple(cfg_blocks),
        cfg_edges=tuple(cfg_edges),
        dfg_edges=tuple(dfg_edges),
    )


def cfg_blocks_compute(cfg_dfg_analysis: _CfgDfgAnalysis) -> TabularFrame:
    """Build CFG blocks from parsed AST inputs.

    Returns
    -------
    polars.LazyFrame
        Lazy frame of CFG block rows.
    """
    if not cfg_dfg_analysis.cfg_blocks:
        return empty_frame_for_table(CFG_BLOCKS_TABLE_KEY)
    frame = pl.DataFrame([dataclasses.asdict(row) for row in cfg_dfg_analysis.cfg_blocks])
    return frame.lazy().select(
        [
            "function_goid_h128",
            "block_idx",
            "block_id",
            "label",
            "file_path",
            "start_line",
            "end_line",
            "kind",
            "stmts_json",
            "in_degree",
            "out_degree",
        ]
    )


def cfg_edges_compute(cfg_dfg_analysis: _CfgDfgAnalysis) -> InferableTabularInput:
    """Build CFG edges from parsed AST inputs.

    Returns
    -------
    InferableTabularInput
        Tabular input of CFG edge rows.
    """
    if not cfg_dfg_analysis.cfg_edges:
        return empty_reader_for_table(CFG_EDGES_TABLE_KEY)
    rows = (dataclasses.asdict(row) for row in cfg_dfg_analysis.cfg_edges)
    reader, _ = record_batch_reader_for_rows(CFG_EDGES_TABLE_KEY, rows)
    return reader


def dfg_edges_compute(cfg_dfg_analysis: _CfgDfgAnalysis) -> InferableTabularInput:
    """Build DFG edges from parsed AST inputs.

    Returns
    -------
    InferableTabularInput
        Tabular input of DFG edge rows.
    """
    if not cfg_dfg_analysis.dfg_edges:
        return empty_reader_for_table(DFG_EDGES_TABLE_KEY)
    rows = (dataclasses.asdict(row) for row in cfg_dfg_analysis.dfg_edges)
    reader, _ = record_batch_reader_for_rows(DFG_EDGES_TABLE_KEY, rows)
    return reader


def cfg_blocks_existing(env: BuildEnv) -> InferableTabularInput:
    """Load CFG blocks from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing CFG blocks.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=CFG_BLOCKS_TABLE_KEY,
        snapshot_id=env.commit,
    )


def cfg_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load CFG edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing CFG edges.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=CFG_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def dfg_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load DFG edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing DFG edges.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=DFG_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def cfg_blocks_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for CFG blocks.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for CFG blocks.
    """
    _ = env
    return empty_reader_for_table(CFG_BLOCKS_TABLE_KEY)


def cfg_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for CFG edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for CFG edges.
    """
    _ = env
    return empty_reader_for_table(CFG_EDGES_TABLE_KEY)


def dfg_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for DFG edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for DFG edges.
    """
    _ = env
    return empty_reader_for_table(DFG_EDGES_TABLE_KEY)


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
