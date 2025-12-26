"""Native Hamilton implementation for call_graph target.

This module implements call graph construction as a native Hamilton pipeline with:
- t__call_graph__run: Parse source files and collect call edges
- t__call_graph__ingest: Package row payloads for materialization
- t__call_graph: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import ibis
import ibis.expr.types as ir
import libcst as cst

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.helpers import filter_paths, get_source_root
from codeintel.build.hamilton.native.options.graphs import CallGraphOptions
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
from codeintel.build.hamilton.run_records import TargetRunRecord, options_hash_for_target
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import FunctionSpanIndex, load_function_index
from codeintel.core.ibis_typing import and_predicates, filter_by, isin_values
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.generated_rows.graph import (
    GraphCallGraphNodesRow as CallGraphNodeRow,
)
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.graphs.compute.callgraph import (
    EdgeResolutionContext,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
)
from codeintel.graphs.compute.callgraph.persistence import dedupe_edge_rows
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.core.schemas.generated_rows.graph import (
        GraphCallGraphEdgesRow as CallGraphEdgeRow,
    )
log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

CALL_GRAPH_TARGET_NAME = "call_graph"
CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"
CALL_GRAPH_TABLE_KEYS = (
    CALL_GRAPH_NODES_TABLE_KEY,
    CALL_GRAPH_EDGES_TABLE_KEY,
)

CALL_GRAPH_SAVE_CONTEXT = SaverContext(
    domain="graphs",
    target=CALL_GRAPH_TARGET_NAME,
    hash_options_node="call_graph__hash_options",
)


@dataclass(frozen=True)
class CallGraphToolOutput(ToolStepOutput):
    """Tool step output for call graph extraction."""

    node_rows: tuple[tuple[object, ...], ...] = ()
    edge_rows: tuple[tuple[object, ...], ...] = ()


@dataclass(frozen=True)
class CallGraphRunInputs:
    """Inputs required for call graph execution."""

    modules: ir.Table
    goids: ir.Table
    goids_record: TargetRunRecord
    source_root: Path | None
    function_index: FunctionSpanIndex | None


@tag_helper(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def call_graph__hash_options(
    env: BuildEnv,
    goids__hash_options: InputHashOptions,
) -> InputHashOptions:
    """Build hash options for call graph materialization.

    Returns
    -------
    InputHashOptions
        Return value.

    """
    options_hash = options_hash_for_target(env, CALL_GRAPH_TARGET_NAME)
    return InputHashOptions(
        options_hash=options_hash,
        manifests=env.manifest_index,
        file_state_hash=goids__hash_options.file_state_hash,
    )


@tag_helper(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def call_graph__source_root(env: BuildEnv) -> Path | None:
    """Resolve repository root for call graph extraction.

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


@tag_helper(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def call_graph__function_index(env: BuildEnv) -> FunctionSpanIndex | None:
    """Load the function index for call graph resolution.

    Returns
    -------
    FunctionSpanIndex | None
        Return value.

    """
    try:
        return load_function_index(env.gateway, repo=env.snapshot.repo, commit=env.snapshot.commit)
    except (OSError, RuntimeError, ValueError) as exc:
        log.warning("call_graph: Failed to load function index: %s", exc)
        return None


@tag_helper(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def call_graph__run_inputs(
    q__core__modules: ir.Table,
    q__core__goids: ir.Table,
    t__goids: TargetRunRecord,
    call_graph__source_root: Path | None,
    call_graph__function_index: FunctionSpanIndex | None,
) -> CallGraphRunInputs:
    """Bundle inputs for call graph execution.

    Returns
    -------
    CallGraphRunInputs
        Return value.

    """
    return CallGraphRunInputs(
        modules=q__core__modules,
        goids=q__core__goids,
        goids_record=t__goids,
        source_root=call_graph__source_root,
        function_index=call_graph__function_index,
    )


@dataclass(frozen=True)
class _EdgeCollectionState:
    """State for edge collection across files.

    Attributes
    ----------
    function_index
        Function span index for GOID lookup.
    global_callees
        Global qualname to GOID mapping.
    def_goids_by_path
        Module GOIDs by path.
    source_root
        Repository root path.
    repo
        Repository identifier.
    commit
        Commit SHA.
    use_libcst
        Whether to use LibCST for parsing.
    resolve_imports
        Whether to resolve import aliases.
    max_edges_per_file
        Maximum edges per file (0=unlimited).
    """

    function_index: FunctionSpanIndex
    global_callees: dict[str, int]
    def_goids_by_path: dict[str, int]
    source_root: Path
    repo: str
    commit: str
    use_libcst: bool
    resolve_imports: bool
    max_edges_per_file: int


@tag_helper(domain="graphs")
def _log_repo_state(
    q__core__modules: ir.Table,
    q__core__goids: ir.Table,
    repo: str,
    commit: str,
) -> None:
    """Log current module/GOID counts to aid validation diagnostics.

    Parameters
    ----------
    q__core__modules
        Ibis table expression for core.modules.
    q__core__goids
        Ibis table expression for core.goids.
    repo
        Repository identifier.
    commit
        Commit SHA.
    """
    try:
        modules_tbl = q__core__modules
        goids_tbl = q__core__goids

        module_count = int(
            cast(
                "SupportsInt",
                modules_tbl.filter(
                    and_predicates(modules_tbl.repo == repo, modules_tbl.commit == commit)
                )
                .count()
                .execute(),
            )
        )
        goid_count = int(
            cast(
                "SupportsInt",
                goids_tbl.filter(and_predicates(goids_tbl.repo == repo, goids_tbl.commit == commit))
                .count()
                .execute(),
            )
        )
        module_goid_count = int(
            cast(
                "SupportsInt",
                goids_tbl.filter(
                    and_predicates(
                        goids_tbl.repo == repo,
                        goids_tbl.commit == commit,
                        goids_tbl.kind == "module",
                    )
                )
                .count()
                .execute(),
            )
        )
        log.info(
            "call_graph repo_state modules=%d goids=%d (module_kind=%d)",
            module_count,
            goid_count,
            module_goid_count,
        )
    except DuckDBError as exc:
        log.debug("call_graph: Could not query repo state: %s", exc)


@tag_helper(domain="graphs")
def _build_global_callee_lookup(
    q__core__goids: ir.Table,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Build a lookup mapping qualnames to function GOIDs.

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
    dict[str, int]
        Mapping of qualname to GOID.
    """
    try:
        goids_tbl = q__core__goids
        expr = (
            filter_by(
                goids_tbl,
                goids_tbl.repo == repo,
                goids_tbl.commit == commit,
                isin_values(goids_tbl.kind, ["function", "method"]),
            )
            .select(goids_tbl.qualname, goids_tbl.goid_h128)
            .order_by(goids_tbl.qualname)
        )
        rows = expr.execute()
        return {
            str(qualname): int(goid) for qualname, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper(domain="graphs")
def _build_def_goids_by_path(
    q__core__goids: ir.Table,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Build lookup of module GOIDs by path.

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
    dict[str, int]
        Mapping of relative path to module GOID.
    """
    try:
        goids_tbl = q__core__goids
        expr = (
            filter_by(
                goids_tbl,
                goids_tbl.repo == repo,
                goids_tbl.commit == commit,
                goids_tbl.kind == "module",
            )
            .select(goids_tbl.rel_path, goids_tbl.goid_h128)
            .order_by(goids_tbl.rel_path)
        )
        rows = expr.execute()
        return {
            normalize_path(str(rel_path)): int(goid)
            for rel_path, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper(domain="graphs")
def _collect_edges_for_file(
    rel_path: str,
    file_path: Path,
    context: EdgeResolutionContext,
    *,
    use_libcst: bool,
    max_edges_per_file: int,
) -> list[CallGraphEdgeRow]:
    """Collect call edges for a single Python file.

    Tries LibCST first, falls back to AST on parse failures.

    Parameters
    ----------
    rel_path
        Relative path of the file.
    file_path
        Absolute path to the file.
    context
        Resolution context with function index and callee maps.
    use_libcst
        Whether to prefer LibCST before AST fallback.
    max_edges_per_file
        Maximum number of edges to retain per file (0 = unlimited).

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected call graph edges.
    """
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf8")
    except (OSError, UnicodeDecodeError):
        return []

    if not use_libcst:
        edges = collect_edges_ast(rel_path, file_path, context)
    else:
        try:
            module = cst.parse_module(source)
            edges = collect_edges_cst(rel_path, module, context)
        except cst.ParserSyntaxError:
            edges = collect_edges_ast(rel_path, file_path, context)

    if max_edges_per_file > 0 and len(edges) > max_edges_per_file:
        return edges[:max_edges_per_file]
    return edges


@tag_helper(domain="graphs")
def _build_nodes_from_goids(
    q__core__goids: ir.Table,
    repo: str,
    commit: str,
) -> list[CallGraphNodeRow]:
    """Build call graph node rows from function GOIDs.

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
    list[CallGraphNodeRow]
        Node rows for all functions.
    """
    try:
        goids_tbl = q__core__goids

        language_expr = (
            goids_tbl.language if "language" in goids_tbl.columns else ibis.literal("python")
        )
        rel_path_expr = goids_tbl.rel_path if "rel_path" in goids_tbl.columns else ibis.literal("")
        kind_expr = goids_tbl.kind if "kind" in goids_tbl.columns else ibis.literal("function")

        expr = (
            filter_by(
                goids_tbl,
                goids_tbl.repo == repo,
                goids_tbl.commit == commit,
                isin_values(goids_tbl.kind, ["function", "method"]),
            )
            .select(
                goids_tbl.goid_h128,
                ibis.coalesce(language_expr, ibis.literal("python")).name("language"),
                kind_expr.name("kind"),
                rel_path_expr.name("rel_path"),
            )
            .order_by(goids_tbl.goid_h128)
        )
        rows = expr.execute()
    except DuckDBError as exc:
        log.debug("call_graph: Could not build nodes from GOIDs: %s", exc)
        return []

    return [
        CallGraphNodeRow(
            goid_h128=int(goid_h128),
            language=str(language) if language else "python",
            kind=str(kind),
            arity=0,
            is_public=True,
            rel_path=str(rel_path),
        )
        for goid_h128, language, kind, rel_path in rows.itertuples(index=False, name=None)
    ]


@tag_helper(domain="graphs")
def _collect_all_edges(
    paths: list[str],
    ctx: _EdgeCollectionState,
) -> list[CallGraphEdgeRow]:
    """Collect edges from all files with functions.

    Parameters
    ----------
    paths
        List of relative paths containing functions.
    ctx
        Edge collection context with lookups and config.

    Returns
    -------
    list[CallGraphEdgeRow]
        All collected edges.
    """
    all_edges: list[CallGraphEdgeRow] = []

    for rel_path in paths:
        local_callees = ctx.function_index.local_name_map(rel_path)
        file_path = ctx.source_root / rel_path
        import_aliases: dict[str, str] = {}
        if ctx.resolve_imports and file_path.exists():
            with contextlib.suppress(OSError, UnicodeDecodeError, cst.ParserSyntaxError):
                import_aliases = collect_aliases(
                    cst.parse_module(file_path.read_text(encoding="utf8"))
                )

        context = EdgeResolutionContext(
            repo=ctx.repo,
            commit=ctx.commit,
            function_index=ctx.function_index,
            local_callees=local_callees,
            global_callees=ctx.global_callees,
            import_aliases=import_aliases,
            scip_candidates_by_use_path={},
            def_goids_by_path=ctx.def_goids_by_path,
        )
        all_edges.extend(
            _collect_edges_for_file(
                rel_path,
                file_path,
                context,
                use_libcst=ctx.use_libcst,
                max_edges_per_file=ctx.max_edges_per_file,
            )
        )

    return all_edges


def _serialize_edge_row(edge: CallGraphEdgeRow) -> CallGraphEdgeRow:
    evidence = edge["evidence_json"]
    if isinstance(evidence, dict):
        return {**edge, "evidence_json": json.dumps(evidence)}
    return edge


def _serialize_call_graph_edges(edges: list[CallGraphEdgeRow]) -> list[CallGraphEdgeRow]:
    return [_serialize_edge_row(edge) for edge in dedupe_edge_rows(edges)]


def _coerce_call_graph_output(output: ToolStepOutput) -> CallGraphToolOutput:
    if isinstance(output, CallGraphToolOutput):
        return output
    return CallGraphToolOutput(result=output.result)


@tag_tool(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def t__call_graph__run(
    env: BuildEnv,
    graph: TargetGraph,
    call_graph__hash_options: InputHashOptions,
    call_graph__run_inputs: CallGraphRunInputs,
) -> CallGraphToolOutput:
    """Execute call graph extraction on repository modules.

    Returns
    -------
    CallGraphToolOutput
        Return value.

    """
    context = ToolRunContext(
        env=env,
        graph=graph,
        target_name=CALL_GRAPH_TARGET_NAME,
        hash_options=call_graph__hash_options,
        skip_reason="call_graph skipped",
    )

    def _execute() -> CallGraphToolOutput:
        if call_graph__run_inputs.goids_record.status != "succeeded":
            return CallGraphToolOutput(
                result=ExecutionResult.failed(
                    f"Upstream goids target failed: {call_graph__run_inputs.goids_record.error}"
                )
            )

        function_index = call_graph__run_inputs.function_index
        if function_index is None:
            return CallGraphToolOutput(
                result=ExecutionResult.failed("call_graph function index is unavailable")
            )

        source_root = call_graph__run_inputs.source_root
        if source_root is None:
            return CallGraphToolOutput(
                result=ExecutionResult.failed("call_graph source root could not be resolved")
            )

        opts = load_target_options(
            env,
            target_name=CALL_GRAPH_TARGET_NAME,
            options_type=CallGraphOptions,
        )

        _log_repo_state(
            call_graph__run_inputs.modules,
            call_graph__run_inputs.goids,
            env.snapshot.repo,
            env.snapshot.commit,
        )

        paths = filter_paths(function_index.paths(), scope_paths=opts.scope_paths)
        if not paths:
            return CallGraphToolOutput(
                result=ExecutionResult.ok(
                    table_counts={
                        CALL_GRAPH_NODES_TABLE_KEY: 0,
                        CALL_GRAPH_EDGES_TABLE_KEY: 0,
                    }
                )
            )

        collection_ctx = _EdgeCollectionState(
            function_index=function_index,
            global_callees=_build_global_callee_lookup(
                call_graph__run_inputs.goids,
                env.snapshot.repo,
                env.snapshot.commit,
            ),
            def_goids_by_path=_build_def_goids_by_path(
                call_graph__run_inputs.goids,
                env.snapshot.repo,
                env.snapshot.commit,
            ),
            source_root=source_root,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            use_libcst=opts.use_libcst,
            resolve_imports=opts.resolve_imports,
            max_edges_per_file=opts.max_edges_per_file,
        )
        edges = _collect_all_edges(paths, collection_ctx)
        log.info("call_graph: Collected %d edges from %d files", len(edges), len(paths))

        node_rows = tuple(
            row_to_tuple(CALL_GRAPH_NODES_TABLE_KEY, row)
            for row in _build_nodes_from_goids(
                call_graph__run_inputs.goids,
                env.snapshot.repo,
                env.snapshot.commit,
            )
        )
        edge_rows = tuple(
            row_to_tuple(CALL_GRAPH_EDGES_TABLE_KEY, row)
            for row in _serialize_call_graph_edges(edges)
        )

        log.info("call_graph: Built %d nodes, %d edges", len(node_rows), len(edge_rows))

        return CallGraphToolOutput(
            result=ExecutionResult.ok(
                table_counts={
                    CALL_GRAPH_NODES_TABLE_KEY: len(node_rows),
                    CALL_GRAPH_EDGES_TABLE_KEY: len(edge_rows),
                }
            ),
            node_rows=node_rows,
            edge_rows=edge_rows,
        )

    return _coerce_call_graph_output(run_tool_step(context=context, run=_execute))


@tag_compute(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def t__call_graph__ingest(
    t__call_graph__run: CallGraphToolOutput,
) -> IngestStep[dict[str, tuple[tuple[object, ...], ...]]]:
    """Package call graph rows for table materialization.

    Returns
    -------
    IngestStep[dict[str, tuple[tuple[object, ...], ...]]]
        Return value.

    """
    result = t__call_graph__run.result
    if result.skipped:
        return IngestStep(
            result=ExecutionResult.skip(
                result.skip_reason or "call_graph skipped",
                warnings=result.warnings,
            )
        )
    if not result.success:
        return IngestStep(
            result=ExecutionResult.failed(
                result.error or "call_graph failed",
                warnings=result.warnings,
            )
        )
    payload = {
        CALL_GRAPH_NODES_TABLE_KEY: t__call_graph__run.node_rows,
        CALL_GRAPH_EDGES_TABLE_KEY: t__call_graph__run.edge_rows,
    }
    table_counts = {
        CALL_GRAPH_NODES_TABLE_KEY: len(t__call_graph__run.node_rows),
        CALL_GRAPH_EDGES_TABLE_KEY: len(t__call_graph__run.edge_rows),
    }
    return IngestStep(
        result=ExecutionResult.ok(table_counts=table_counts, warnings=result.warnings),
        payload=payload,
    )


@save_rows(
    context=CALL_GRAPH_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CALL_GRAPH_NODES_TABLE_KEY),
)
@tag_compute(domain="graphs", target=CALL_GRAPH_TARGET_NAME, target_="call_graph__nodes_rows")
def call_graph__nodes_rows(
    t__call_graph__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.call_graph_nodes.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__call_graph__ingest.result.skipped or not t__call_graph__ingest.result.success:
        return None
    payload = t__call_graph__ingest.payload
    if payload is None:
        msg = "Missing call_graph ingest payload"
        raise ValueError(msg)
    rows = payload.get(CALL_GRAPH_NODES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {CALL_GRAPH_NODES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@save_rows(
    context=CALL_GRAPH_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=CALL_GRAPH_EDGES_TABLE_KEY),
)
@tag_compute(domain="graphs", target=CALL_GRAPH_TARGET_NAME, target_="call_graph__edges_rows")
def call_graph__edges_rows(
    t__call_graph__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for graph.call_graph_edges.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Return value.

    Raises
    ------
    ValueError
        If the ingest payload or rows are missing.

    """
    if t__call_graph__ingest.result.skipped or not t__call_graph__ingest.result.success:
        return None
    payload = t__call_graph__ingest.payload
    if payload is None:
        msg = "Missing call_graph ingest payload"
        raise ValueError(msg)
    rows = payload.get(CALL_GRAPH_EDGES_TABLE_KEY)
    if rows is None:
        msg = f"Missing rows for {CALL_GRAPH_EDGES_TABLE_KEY}"
        raise ValueError(msg)
    return rows


@tag_helper(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def call_graph__table_materializations(
    m__graph__call_graph_nodes: MaterializationMetadata,
    m__graph__call_graph_edges: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect materialization metadata for call graph tables.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Return value.

    """
    return {
        CALL_GRAPH_NODES_TABLE_KEY: m__graph__call_graph_nodes,
        CALL_GRAPH_EDGES_TABLE_KEY: m__graph__call_graph_edges,
    }


@tag_helper(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def call_graph__finalize_context(
    env: BuildEnv,
    graph: TargetGraph,
    call_graph__hash_options: InputHashOptions,
) -> ToolFinalizeContext:
    """Build finalization context for call graph.

    Returns
    -------
    ToolFinalizeContext
        Return value.

    """
    return ToolFinalizeContext(
        env=env,
        graph=graph,
        target_name=CALL_GRAPH_TARGET_NAME,
        hash_options=call_graph__hash_options,
    )


@codeintel_target(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def t__call_graph(
    call_graph__finalize_context: ToolFinalizeContext,
    t__call_graph__run: CallGraphToolOutput,
    t__call_graph__ingest: IngestStep[dict[str, tuple[tuple[object, ...], ...]]],
    call_graph__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Construct a function call graph.

    Returns
    -------
    TargetRunRecord
        Return value.

    """
    return finalize_target_from_materializations(
        context=call_graph__finalize_context,
        tool_step=t__call_graph__run,
        ingest_step=t__call_graph__ingest,
        artifact_materializations=None,
        table_materializations=call_graph__table_materializations,
    )


__all__ = [
    "t__call_graph",
    "t__call_graph__ingest",
    "t__call_graph__run",
]
