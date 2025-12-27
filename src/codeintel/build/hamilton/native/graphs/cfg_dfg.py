"""Native Hamilton implementation for cfg and dfg targets.

This module implements CFG (Control Flow Graph) and DFG (Data Flow Graph)
construction as native Hamilton pipelines with:
- t__cfg__run: Parse functions and build CFG blocks/edges
- t__cfg: Materialize CFG target via shared saver helpers
- t__dfg__run: Build DFG edges from CFG results
- t__dfg: Materialize DFG target via shared saver helpers

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ibis.expr.types as ir

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.helpers import filter_paths, get_source_root
from codeintel.build.hamilton.native.options.graphs import CfgDfgOptions
from codeintel.build.hamilton.native.patterns import (
    IngestStep,
    SaverContext,
    TableSaveSpec,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.core.ibis_typing import filter_by, isin_values
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import cfg as cfg_compute
from codeintel.graphs.compute import dfg as dfg_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.core.data_models.rows import CFGBlockRow, CFGEdgeRow, DFGEdgeRow
log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord)

CFG_TARGET_NAME = "cfg"
DFG_TARGET_NAME = "dfg"

CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
CFG_EDGES_TABLE_KEY = "graph.cfg_edges"
DFG_EDGES_TABLE_KEY = "graph.dfg_edges"

CFG_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=CFG_TARGET_NAME,
)
DFG_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=DFG_TARGET_NAME,
)


@dataclass(frozen=True)
class CfgToolOutput(ToolStepOutput):
    """Tool step output for CFG extraction."""

    cfg_results: tuple[cfg_compute.CFGResult, ...] = ()
    block_rows: tuple[CFGBlockRow, ...] = ()
    edge_rows: tuple[CFGEdgeRow, ...] = ()


@dataclass(frozen=True)
class DfgToolOutput(ToolStepOutput):
    """Tool step output for DFG extraction."""

    edge_rows: tuple[DFGEdgeRow, ...] = ()


@dataclass(frozen=True)
class CfgRunInputs:
    """Inputs required for CFG extraction."""

    goids: ir.Table
    goids_record: TargetRunRecord
    source_root: Path | None






@tag_helper(domain="graphs", target=CFG_TARGET_NAME)
def cfg__source_root(env: BuildEnv) -> Path | None:
    """Resolve repository root for CFG extraction.

    Returns
    -------
    Path | None
        Return value.

    """
    repo_root = env.snapshot.repo_root
    if repo_root is not None:
        return repo_root
    try:
        return get_source_root(env.gateway, env.snapshot.repo, env.snapshot.commit)
    except (OSError, RuntimeError, ValueError):
        return None


@tag_helper(domain="graphs", target=CFG_TARGET_NAME)
def cfg__run_inputs(
    q__core__goids: ir.Table,
    t__goids: TargetRunRecord,
    cfg__source_root: Path | None,
) -> CfgRunInputs:
    """Bundle inputs for CFG extraction.

    Returns
    -------
    CfgRunInputs
        Return value.

    """
    return CfgRunInputs(
        goids=q__core__goids,
        goids_record=t__goids,
        source_root=cfg__source_root,
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


@tag_helper(domain="graphs")
def _load_functions(
    q__core__goids: ir.Table,
    repo: str,
    commit: str,
) -> list[FunctionInfo]:
    """Load function metadata from GOIDs table.

    Parameters
    ----------
    q__core__goids
        Ibis table expression for core.goids.
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
        expr = filter_by(
            q__core__goids,
            q__core__goids.repo == repo,
            q__core__goids.commit == commit,
            isin_values(q__core__goids.kind, ["function", "method"]),
        ).select(
            q__core__goids.goid_h128,
            q__core__goids.qualname,
            q__core__goids.rel_path,
            q__core__goids.start_line,
            q__core__goids.end_line,
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


@tag_helper(domain="graphs")
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


@tag_helper(domain="graphs")
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


@tag_helper(domain="graphs")
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


def _filter_functions_for_scope(
    functions: list[FunctionInfo],
    *,
    scope_paths: list[str] | None,
) -> list[FunctionInfo]:
    paths = list({f.rel_path for f in functions})
    filtered_paths = set(filter_paths(paths, scope_paths=scope_paths))
    return [function for function in functions if function.rel_path in filtered_paths]


def _coerce_cfg_output(output: ToolStepOutput) -> CfgToolOutput:
    if isinstance(output, CfgToolOutput):
        return output
    return CfgToolOutput(result=output.result)


def _coerce_dfg_output(output: ToolStepOutput) -> DfgToolOutput:
    if isinstance(output, DfgToolOutput):
        return output
    return DfgToolOutput(result=output.result)


@tag_tool(domain="graphs", target=CFG_TARGET_NAME)
def t__cfg__run(
    env: BuildEnv,
    catalog: DagCatalog,
    cfg__run_inputs: CfgRunInputs,
) -> CfgToolOutput:
    """Execute CFG extraction for all functions.

    Returns
    -------
    CfgToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=CFG_TARGET_NAME,
    )

    def _execute() -> CfgToolOutput:
        if cfg__run_inputs.goids_record.status == "skipped":
            return CfgToolOutput(result=ExecutionResult.skip("Upstream goids target skipped"))
        if cfg__run_inputs.goids_record.status != "succeeded":
            return CfgToolOutput(
                result=ExecutionResult.failed(
                    f"Upstream goids target failed: {cfg__run_inputs.goids_record.error}"
                )
            )

        source_root = cfg__run_inputs.source_root
        if source_root is None:
            return CfgToolOutput(
                result=ExecutionResult.failed("CFG source root could not be resolved")
            )

        opts = load_target_options(env, target_name=CFG_TARGET_NAME, options_type=CfgDfgOptions)
        functions = _filter_functions_for_scope(
            _load_functions(cfg__run_inputs.goids, env.snapshot.repo, env.snapshot.commit),
            scope_paths=opts.scope_paths,
        )

        if not functions:
            return CfgToolOutput(
                result=ExecutionResult.ok(
                    table_counts={
                        CFG_BLOCKS_TABLE_KEY: 0,
                        CFG_EDGES_TABLE_KEY: 0,
                    }
                ),
                cfg_results=(),
                block_rows=(),
                edge_rows=(),
            )

        cfg_results, block_rows, edge_rows = _process_all_files(functions, source_root)
        log.info(
            "cfg: Built %d blocks and %d edges for %d functions",
            len(block_rows),
            len(edge_rows),
            len(functions),
        )
        return CfgToolOutput(
            result=ExecutionResult.ok(
                table_counts={
                    CFG_BLOCKS_TABLE_KEY: len(block_rows),
                    CFG_EDGES_TABLE_KEY: len(edge_rows),
                }
            ),
            cfg_results=tuple(cfg_results),
            block_rows=tuple(block_rows),
            edge_rows=tuple(edge_rows),
        )

    return _coerce_cfg_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=CFG_TARGET_NAME)
def t__cfg__ingest(
    t__cfg__run: CfgToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package CFG rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__cfg__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "cfg skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "cfg failed",
                warnings=result.warnings,
            )
        )

    block_rows = tuple(row.to_tuple() for row in t__cfg__run.block_rows)
    edge_rows = tuple(row.to_tuple() for row in t__cfg__run.edge_rows)
    payload = {
        CFG_BLOCKS_TABLE_KEY: block_rows,
        CFG_EDGES_TABLE_KEY: edge_rows,
    }
    table_counts = {
        CFG_BLOCKS_TABLE_KEY: len(block_rows),
        CFG_EDGES_TABLE_KEY: len(edge_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(context=CFG_SAVE_CONTEXT, spec=TableSaveSpec(table_key=CFG_BLOCKS_TABLE_KEY))
@tag_compute(domain="graphs", target=CFG_TARGET_NAME, target_="cfg__blocks_rows")
def cfg__blocks_rows(
    t__cfg__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.cfg_blocks.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__cfg__ingest.result.skipped or not t__cfg__ingest.result.success:
        return None
    payload = t__cfg__ingest.payload
    if payload is None:
        msg = "Missing cfg ingest payload"
        raise ValueError(msg)
    rows = payload.get(CFG_BLOCKS_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {CFG_BLOCKS_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(context=CFG_SAVE_CONTEXT, spec=TableSaveSpec(table_key=CFG_EDGES_TABLE_KEY))
@tag_compute(domain="graphs", target=CFG_TARGET_NAME, target_="cfg__edges_rows")
def cfg__edges_rows(
    t__cfg__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.cfg_edges.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__cfg__ingest.result.skipped or not t__cfg__ingest.result.success:
        return None
    payload = t__cfg__ingest.payload
    if payload is None:
        msg = "Missing cfg ingest payload"
        raise ValueError(msg)
    rows = payload.get(CFG_EDGES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {CFG_EDGES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="graphs", target=CFG_TARGET_NAME)
def cfg__table_materializations(
    m__graph__cfg_blocks: MaterializationResult,
    m__graph__cfg_edges: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect materialization results for CFG tables.

    Returns
    -------
    dict[str, MaterializationResult]
        Return value.

    """
    return {
        CFG_BLOCKS_TABLE_KEY: m__graph__cfg_blocks,
        CFG_EDGES_TABLE_KEY: m__graph__cfg_edges,
    }


@tag_helper(domain="graphs", target=CFG_TARGET_NAME)
def cfg__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for CFG.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=CFG_TARGET_NAME,
    )


@codeintel_target(domain="graphs", target=CFG_TARGET_NAME)
def t__cfg(
    cfg__finalize_context: ToolFinalizeContext,
    t__cfg__run: CfgToolOutput,
    t__cfg__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    cfg__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Construct control flow graphs per function.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=cfg__finalize_context,
        tool_step=t__cfg__run,
        ingest_step=t__cfg__ingest,
        artifact_materializations=None,
        table_materializations=cfg__table_materializations,
    )


@tag_tool(domain="graphs", target=DFG_TARGET_NAME)
def t__dfg__run(
    env: BuildEnv,
    catalog: DagCatalog,
    t__cfg__run: CfgToolOutput,
) -> DfgToolOutput:
    """Execute DFG extraction from CFG results.

    Returns
    -------
    DfgToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=DFG_TARGET_NAME,
    )

    def _execute() -> DfgToolOutput:
        cfg_result = t__cfg__run.result
        if cfg_result.skipped:
            return DfgToolOutput(
                result=ExecutionResult.skip(
                    cfg_result.skip_reason or "cfg skipped",
                    warnings=cfg_result.warnings,
                )
            )
        if not cfg_result.success:
            return DfgToolOutput(
                result=ExecutionResult.failed(
                    cfg_result.error or "cfg failed",
                    warnings=cfg_result.warnings,
                )
            )

        cfg_results = t__cfg__run.cfg_results
        if not cfg_results:
            return DfgToolOutput(
                result=ExecutionResult.ok(table_counts={DFG_EDGES_TABLE_KEY: 0}),
                edge_rows=(),
            )

        all_dfg_rows: list[DFGEdgeRow] = []
        for cfg_result_item in cfg_results:
            dfg_result = dfg_compute.build_dfg(
                cfg_result_item.function_goid,
                cfg_result_item.blocks,
                cfg_result_item.edges,
            )
            all_dfg_rows.extend(dfg_compute.dfg_to_rows(dfg_result))

        log.info("dfg: Built %d edges from %d CFGs", len(all_dfg_rows), len(cfg_results))
        return DfgToolOutput(
            result=ExecutionResult.ok(table_counts={DFG_EDGES_TABLE_KEY: len(all_dfg_rows)}),
            edge_rows=tuple(all_dfg_rows),
        )

    return _coerce_dfg_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=DFG_TARGET_NAME)
def t__dfg__ingest(
    t__dfg__run: DfgToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package DFG rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__dfg__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "dfg skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "dfg failed",
                warnings=result.warnings,
            )
        )

    edge_rows = tuple(row.to_tuple() for row in t__dfg__run.edge_rows)
    payload = {DFG_EDGES_TABLE_KEY: edge_rows}
    table_counts = {DFG_EDGES_TABLE_KEY: len(edge_rows)}
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(context=DFG_SAVE_CONTEXT, spec=TableSaveSpec(table_key=DFG_EDGES_TABLE_KEY))
@tag_compute(domain="graphs", target=DFG_TARGET_NAME, target_="dfg__edges_rows")
def dfg__edges_rows(
    t__dfg__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.dfg_edges.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__dfg__ingest.result.skipped or not t__dfg__ingest.result.success:
        return None
    payload = t__dfg__ingest.payload
    if payload is None:
        msg = "Missing dfg ingest payload"
        raise ValueError(msg)
    rows = payload.get(DFG_EDGES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {DFG_EDGES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="graphs", target=DFG_TARGET_NAME)
def dfg__table_materializations(
    m__graph__dfg_edges: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect materialization results for DFG tables.

    Returns
    -------
    dict[str, MaterializationResult]
        Return value.

    """
    return {DFG_EDGES_TABLE_KEY: m__graph__dfg_edges}


@tag_helper(domain="graphs", target=DFG_TARGET_NAME)
def dfg__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for DFG.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=DFG_TARGET_NAME,
    )


@codeintel_target(domain="graphs", target=DFG_TARGET_NAME)
def t__dfg(
    dfg__finalize_context: ToolFinalizeContext,
    t__dfg__run: DfgToolOutput,
    t__dfg__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    dfg__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Construct data flow graphs per function.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=dfg__finalize_context,
        tool_step=t__dfg__run,
        ingest_step=t__dfg__ingest,
        artifact_materializations=None,
        table_materializations=dfg__table_materializations,
    )


__all__ = [
    "CfgToolOutput",
    "DfgToolOutput",
    "FunctionInfo",
    "t__cfg",
    "t__cfg__ingest",
    "t__cfg__run",
    "t__dfg",
    "t__dfg__ingest",
    "t__dfg__run",
]
