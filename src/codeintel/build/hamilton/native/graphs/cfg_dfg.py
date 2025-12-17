"""Native Hamilton implementation for cfg and dfg targets.

This module implements CFG (Control Flow Graph) and DFG (Data Flow Graph)
construction as native Hamilton pipelines with:
- t__cfg__extract: Parse functions and build CFG blocks and edges
- t__cfg: Materialize CFG target
- t__dfg__extract: Build DFG edges from CFG results
- t__dfg: Materialize DFG target

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import filter_paths, get_source_root, persist_rows
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.options.graphs import CfgDfgOptions
from codeintel.build.hamilton.native.target_spec_helpers import make_output_target
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.targets import TargetGraph
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import cfg as cfg_compute
from codeintel.graphs.compute import dfg as dfg_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.data_models.rows import CFGBlockRow, CFGEdgeRow, DFGEdgeRow
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

TARGET_SPECS = (
    make_output_target(
        name="cfg",
        module="graphs",
        description="Control flow graph construction per function.",
        table_keys=(
            "graph.cfg_blocks",
            "graph.cfg_edges",
        ),
    ),
    make_output_target(
        name="dfg",
        module="graphs",
        description="Data flow graph construction per function.",
        table_keys=("graph.dfg_edges",),
    ),
)


@dataclass(frozen=True)
class FunctionInfo:
    """Information about a function for CFG/DFG construction.

    Attributes
    ----------
    goid
        Function GOID.
    qualname
        Fully qualified name.
    rel_path
        Relative file path.
    start_line
        Starting line number.
    end_line
        Ending line number.
    """

    goid: int
    qualname: str
    rel_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CFGExtractResult:
    """Result from CFG extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    block_count
        Number of CFG blocks extracted.
    edge_count
        Number of CFG edges extracted.
    cfg_results
        List of CFG results for DFG construction.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    block_count: int = 0
    edge_count: int = 0
    cfg_results: list[cfg_compute.CFGResult] = field(default_factory=list)
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class DFGExtractResult:
    """Result from DFG extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    edge_count
        Number of DFG edges extracted.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    edge_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def _load_functions(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[FunctionInfo]:
    """Load function metadata from GOIDs table.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    list[FunctionInfo]
        Function information for CFG construction.
    """
    try:
        goids_tbl = gateway.ibis.table("core.goids")
        expr = goids_tbl.filter(
            cast("Any", goids_tbl.repo == repo)
            & cast("Any", goids_tbl.commit == commit)
            & cast("Any", goids_tbl.kind.isin(cast("Any", ["function", "method"])))
        ).select(
            goids_tbl.goid_h128,
            goids_tbl.qualname,
            goids_tbl.rel_path,
            goids_tbl.start_line,
            goids_tbl.end_line,
        )
        rows = expr.execute()

        return [
            FunctionInfo(
                goid=int(goid),
                qualname=str(qualname),
                rel_path=normalize_path(str(rel_path)),
                start_line=int(start or 1),
                end_line=int(end or start or 1),
            )
            for goid, qualname, rel_path, start, end in rows.itertuples(index=False, name=None)
        ]
    except DuckDBError:
        return []


@tag(node_type="helper")
def _parse_file_functions(
    file_path: Path,
) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Parse a file and extract function definitions.

    Parameters
    ----------
    file_path
        Absolute path to the file.

    Returns
    -------
    list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]
        List of (qualname_suffix, function_node) tuples.
    """
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def _visit(node: ast.AST, prefix: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{prefix}.{node.name}" if prefix else node.name
            functions.append((qualname, node))
            for child in ast.iter_child_nodes(node):
                _visit(child, qualname)
        elif isinstance(node, ast.ClassDef):
            class_prefix = f"{prefix}.{node.name}" if prefix else node.name
            for child in ast.iter_child_nodes(node):
                _visit(child, class_prefix)
        else:
            for child in ast.iter_child_nodes(node):
                _visit(child, prefix)

    for child in ast.iter_child_nodes(tree):
        _visit(child, "")

    return functions


@tag(node_type="helper")
def _build_cfg_dfg_for_function(
    goid: int,
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    rel_path: str,
    start_line: int,
    end_line: int,
) -> tuple[cfg_compute.CFGResult, list[CFGBlockRow], list[CFGEdgeRow]]:
    """Build CFG for a single function.

    Parameters
    ----------
    goid
        Function GOID.
    func_node
        Function AST node.
    rel_path
        Relative file path.
    start_line
        Function start line.
    end_line
        Function end line.

    Returns
    -------
    tuple[cfg_compute.CFGResult, list[CFGBlockRow], list[CFGEdgeRow]]
        CFG result, block rows, and edge rows.
    """
    cfg_result = cfg_compute.build_cfg(goid, func_node, rel_path)
    block_rows, edge_rows = cfg_compute.cfg_to_rows(cfg_result, rel_path, start_line, end_line)
    return cfg_result, list(block_rows), list(edge_rows)


@tag(node_type="helper")
def _process_all_files(
    functions: list[FunctionInfo],
    source_root: Path,
) -> tuple[list[cfg_compute.CFGResult], list[CFGBlockRow], list[CFGEdgeRow]]:
    """Process all files and build CFGs.

    Parameters
    ----------
    functions
        Function information list.
    source_root
        Repository root path.

    Returns
    -------
    tuple[list[cfg_compute.CFGResult], list[CFGBlockRow], list[CFGEdgeRow]]
        All CFG results, block rows, and edge rows.
    """
    by_path: dict[str, list[FunctionInfo]] = {}
    for func in functions:
        by_path.setdefault(func.rel_path, []).append(func)

    all_cfg_results: list[cfg_compute.CFGResult] = []
    all_block_rows: list[CFGBlockRow] = []
    all_edge_rows: list[CFGEdgeRow] = []

    for rel_path, path_functions in by_path.items():
        file_path = source_root / rel_path
        parsed_functions = _parse_file_functions(file_path)

        func_by_suffix = dict(parsed_functions)

        for func_info in path_functions:
            suffix = func_info.qualname.split(".", 1)[-1] if "." in func_info.qualname else ""
            func_node = func_by_suffix.get(suffix)

            if func_node is None:
                for parse_suffix, node in parsed_functions:
                    if func_info.qualname.endswith(parse_suffix):
                        func_node = node
                        break

            if func_node is None:
                continue

            cfg_result, block_rows, edge_rows = _build_cfg_dfg_for_function(
                func_info.goid,
                func_node,
                func_info.rel_path,
                func_info.start_line,
                func_info.end_line,
            )
            all_cfg_results.append(cfg_result)
            all_block_rows.extend(block_rows)
            all_edge_rows.extend(edge_rows)

    return all_cfg_results, all_block_rows, all_edge_rows


@tag(domain="graphs", target="cfg", node_type="tool")
def t__cfg__extract(
    env: BuildEnv,
    t__goids: TargetRunRecord,
) -> CFGExtractResult:
    """Execute CFG extraction for all functions.

    This is the compute node for the cfg target. It loads function metadata,
    parses source files, and builds control-flow graphs.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__goids
        Upstream GOIDs target result (for dependency).

    Returns
    -------
    CFGExtractResult
        Result containing block and edge counts plus CFG results.

    Notes
    -----
    Produces:
    - graph.cfg_blocks: CFG basic blocks
    - graph.cfg_edges: CFG edges
    """
    if t__goids.status != "succeeded":
        return CFGExtractResult(
            success=False,
            error=f"Upstream goids target failed: {t__goids.error}",
        )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = CfgDfgOptions()

        functions = _load_functions(gateway, repo, commit)
        paths = list({f.rel_path for f in functions})
        filtered_paths = set(filter_paths(paths, scope_paths=opts.scope_paths))
        functions = [f for f in functions if f.rel_path in filtered_paths]

        if not functions:
            log.info("cfg: No functions found, skipping")
            return CFGExtractResult(
                success=True,
                block_count=0,
                edge_count=0,
                cfg_results=[],
                table_counts={
                    "graph.cfg_blocks": 0,
                    "graph.cfg_edges": 0,
                },
            )

        source_root = env.snapshot.repo_root or get_source_root(gateway, repo, commit)
        cfg_results, block_rows, edge_rows = _process_all_files(functions, source_root)

        log.info(
            "cfg: Built %d blocks and %d edges for %d functions",
            len(block_rows),
            len(edge_rows),
            len(functions),
        )

        block_count = persist_rows(
            gateway,
            "graph.cfg_blocks",
            block_rows,
            repo=repo,
            commit=commit,
        )
        edge_count = persist_rows(
            gateway,
            "graph.cfg_edges",
            edge_rows,
            repo=repo,
            commit=commit,
        )

        log.info("cfg: Persisted %d blocks and %d edges", block_count, edge_count)

        return CFGExtractResult(
            success=True,
            block_count=block_count,
            edge_count=edge_count,
            cfg_results=cfg_results,
            table_counts={
                "graph.cfg_blocks": block_count,
                "graph.cfg_edges": edge_count,
            },
        )

    except Exception as exc:
        log.exception("CFG extraction failed")
        return CFGExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="cfg", node_type="materialize")
def t__cfg(
    env: BuildEnv,
    graph: TargetGraph,
    t__cfg__extract: CFGExtractResult,
) -> TargetRunRecord:
    """Materialize CFG target with validation.

    This is the entry point for the cfg target. It orchestrates
    CFG extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__cfg__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, "cfg", t__cfg__extract)


@tag(domain="graphs", target="dfg", node_type="tool")
def t__dfg__extract(
    env: BuildEnv,
    t__cfg__extract: CFGExtractResult,
) -> DFGExtractResult:
    """Execute DFG extraction from CFG results.

    This is the compute node for the dfg target. It takes CFG results
    and builds data-flow graphs using reaching definitions analysis.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__cfg__extract
        CFG extraction result (for CFG blocks and edges).

    Returns
    -------
    DFGExtractResult
        Result containing edge count.

    Notes
    -----
    Produces:
    - graph.dfg_edges: Data-flow edges
    """
    if not t__cfg__extract.success:
        return DFGExtractResult(
            success=False,
            error=f"Upstream cfg extraction failed: {t__cfg__extract.error}",
        )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit

        cfg_results = t__cfg__extract.cfg_results

        if not cfg_results:
            log.info("dfg: No CFG results, skipping")
            return DFGExtractResult(
                success=True,
                edge_count=0,
                table_counts={"graph.dfg_edges": 0},
            )

        all_dfg_rows: list[DFGEdgeRow] = []
        for cfg_result in cfg_results:
            dfg_result = dfg_compute.build_dfg(
                cfg_result.function_goid,
                cfg_result.blocks,
                cfg_result.edges,
            )
            all_dfg_rows.extend(dfg_compute.dfg_to_rows(dfg_result))

        log.info("dfg: Built %d edges from %d CFGs", len(all_dfg_rows), len(cfg_results))

        edge_count = persist_rows(
            gateway,
            "graph.dfg_edges",
            all_dfg_rows,
            repo=repo,
            commit=commit,
        )

        log.info("dfg: Persisted %d edges", edge_count)

        return DFGExtractResult(
            success=True,
            edge_count=edge_count,
            table_counts={"graph.dfg_edges": edge_count},
        )

    except Exception as exc:
        log.exception("DFG extraction failed")
        return DFGExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="dfg", node_type="materialize")
def t__dfg(
    env: BuildEnv,
    graph: TargetGraph,
    t__dfg__extract: DFGExtractResult,
) -> TargetRunRecord:
    """Materialize DFG target with validation.

    This is the entry point for the dfg target. It orchestrates
    DFG extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__dfg__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, "dfg", t__dfg__extract)


__all__ = [
    "CFGExtractResult",
    "DFGExtractResult",
    "FunctionInfo",
    "t__cfg",
    "t__cfg__extract",
    "t__dfg",
    "t__dfg__extract",
]
