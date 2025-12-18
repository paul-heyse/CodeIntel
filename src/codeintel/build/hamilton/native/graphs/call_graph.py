"""Native Hamilton implementation for call_graph target.

This module implements call graph construction as a native Hamilton pipeline with:
- t__call_graph__extract: Parse source files and collect call edges
- t__call_graph: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import ibis
import libcst as cst
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult, to_execution_result
from codeintel.build.hamilton.helpers import filter_paths, get_source_root
from codeintel.build.hamilton.native.options.graphs import CallGraphOptions
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.templates import executor_materialize
from codeintel.build.targets import TargetGraph
from codeintel.core.catalog import load_function_index
from codeintel.core.ibis_typing import and_predicates, filter_by, isin_values
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.generated_rows.graph import (
    GraphCallGraphNodesRow as CallGraphNodeRow,
)
from codeintel.graphs.compute.callgraph import (
    EdgeResolutionContext,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
)
from codeintel.graphs.compute.callgraph.persistence import dedupe_edge_rows
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.catalog import FunctionSpanIndex
    from codeintel.core.schemas.generated_rows.graph import (
        GraphCallGraphEdgesRow as CallGraphEdgeRow,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

CALL_GRAPH_TARGET_NAME = "call_graph"
CALL_GRAPH_NODES_TABLE_KEY = "graph.call_graph_nodes"
CALL_GRAPH_EDGES_TABLE_KEY = "graph.call_graph_edges"
CALL_GRAPH_TABLE_KEYS = (
    CALL_GRAPH_NODES_TABLE_KEY,
    CALL_GRAPH_EDGES_TABLE_KEY,
)

TARGET_SPECS = (
    make_output_target(
        name=CALL_GRAPH_TARGET_NAME,
        module="graphs",
        description="Function call graph construction.",
        options=TargetSpecOptions(
            table_keys=CALL_GRAPH_TABLE_KEYS,
        ),
    ),
)


@dataclass(frozen=True)
class CallGraphExtractResult:
    """Result from call graph extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    node_count
        Number of call graph nodes extracted.
    edge_count
        Number of call graph edges extracted.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    node_count: int = 0
    edge_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def call_graph__execution_result(t__call_graph__extract: CallGraphExtractResult) -> ExecutionResult:
    """Convert call_graph extract result to the executor boundary type.

    Returns
    -------
    ExecutionResult
        Canonical execution result.
    """
    return to_execution_result(t__call_graph__extract, default_error="Call graph extraction failed")


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


@tag(node_type="helper")
def _log_repo_state(gateway: StorageGateway, repo: str, commit: str) -> None:
    """Log current module/GOID counts to aid validation diagnostics.

    Parameters
    ----------
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.
    """
    try:
        modules_tbl = gateway.ibis.table("core.modules")
        goids_tbl = gateway.ibis.table("core.goids")

        module_count = int(
            cast(
                "int",
                gateway.ibis.execute_scalar(
                    modules_tbl.filter(
                        and_predicates(modules_tbl.repo == repo, modules_tbl.commit == commit)
                    ).count()
                ),
            )
        )
        goid_count = int(
            cast(
                "int",
                gateway.ibis.execute_scalar(
                    goids_tbl.filter(
                        and_predicates(goids_tbl.repo == repo, goids_tbl.commit == commit)
                    ).count()
                ),
            )
        )
        module_goid_count = int(
            cast(
                "int",
                gateway.ibis.execute_scalar(
                    goids_tbl.filter(
                        and_predicates(
                            goids_tbl.repo == repo,
                            goids_tbl.commit == commit,
                            goids_tbl.kind == "module",
                        )
                    ).count()
                ),
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


@tag(node_type="helper")
def _build_global_callee_lookup(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Build a lookup mapping qualnames to function GOIDs.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
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
        goids_tbl = gateway.ibis.table("core.goids")
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


@tag(node_type="helper")
def _build_def_goids_by_path(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Build lookup of module GOIDs by path.

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
    dict[str, int]
        Mapping of relative path to module GOID.
    """
    try:
        goids_tbl = gateway.ibis.table("core.goids")
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


@tag(node_type="helper")
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


@tag(node_type="helper")
def _build_nodes_from_goids(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[CallGraphNodeRow]:
    """Build call graph node rows from function GOIDs.

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
    list[CallGraphNodeRow]
        Node rows for all functions.
    """
    try:
        goids_tbl = gateway.ibis.table("core.goids")

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


@tag(node_type="helper")
def _persist_nodes(
    gateway: StorageGateway,
    nodes: list[CallGraphNodeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist call graph nodes.

    Parameters
    ----------
    gateway
        Storage gateway.
    nodes
        Node rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of nodes persisted.
    """
    if not nodes:
        return 0

    gateway.policy.ensure_table(CALL_GRAPH_NODES_TABLE_KEY)
    gateway.policy.delete_for_snapshot(CALL_GRAPH_NODES_TABLE_KEY, repo=repo, commit=commit)
    gateway.policy.bulk_insert_mappings(CALL_GRAPH_NODES_TABLE_KEY, nodes)
    return len(nodes)


@tag(node_type="helper")
def _persist_edges(
    gateway: StorageGateway,
    edges: list[CallGraphEdgeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist call graph edges after deduplication.

    Parameters
    ----------
    gateway
        Storage gateway.
    edges
        Edge rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of edges persisted.
    """
    if not edges:
        return 0

    unique_edges = dedupe_edge_rows(edges)

    serialized: list[CallGraphEdgeRow] = []
    for edge in unique_edges:
        evidence = edge["evidence_json"]
        if isinstance(evidence, dict):
            serialized.append({**edge, "evidence_json": json.dumps(evidence)})
        else:
            serialized.append(edge)

    gateway.policy.ensure_table(CALL_GRAPH_EDGES_TABLE_KEY)
    gateway.policy.delete_for_snapshot(CALL_GRAPH_EDGES_TABLE_KEY, repo=repo, commit=commit)
    gateway.policy.bulk_insert_mappings(CALL_GRAPH_EDGES_TABLE_KEY, serialized)
    return len(serialized)


@tag(node_type="helper")
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


@tag(domain="graphs", target=CALL_GRAPH_TARGET_NAME, node_type="tool")
def t__call_graph__extract(
    env: BuildEnv,
    t__goids: TargetRunRecord,
) -> CallGraphExtractResult:
    """Execute call graph extraction on repository modules.

    This is the compute node for the call_graph target. It loads function
    metadata, parses source files to collect call edges using LibCST or AST,
    and builds the call graph.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__goids
        Upstream GOIDs target result (for dependency).

    Returns
    -------
    CallGraphExtractResult
        Result containing node and edge counts.

    Notes
    -----
    Produces:
    - graph.call_graph_nodes: Call graph nodes
    - graph.call_graph_edges: Call graph edges
    """
    if t__goids.status != "succeeded":
        return CallGraphExtractResult(
            success=False,
            error=f"Upstream goids target failed: {t__goids.error}",
        )

    try:
        gateway = env.gateway
        snapshot = env.snapshot
        repo = snapshot.repo
        commit = snapshot.commit
        opts = CallGraphOptions()

        _log_repo_state(gateway, repo, commit)

        function_index = load_function_index(gateway, repo=repo, commit=commit)
        paths = filter_paths(function_index.paths(), scope_paths=opts.scope_paths)

        if not paths:
            log.info("call_graph: No functions found, skipping")
            return CallGraphExtractResult(
                success=True,
                node_count=0,
                edge_count=0,
                table_counts={
                    CALL_GRAPH_NODES_TABLE_KEY: 0,
                    CALL_GRAPH_EDGES_TABLE_KEY: 0,
                },
            )

        global_callees = _build_global_callee_lookup(gateway, repo, commit)
        def_goids = _build_def_goids_by_path(gateway, repo, commit)

        source_root = snapshot.repo_root or get_source_root(gateway, repo, commit)

        collection_ctx = _EdgeCollectionState(
            function_index=function_index,
            global_callees=global_callees,
            def_goids_by_path=def_goids,
            source_root=source_root,
            repo=repo,
            commit=commit,
            use_libcst=opts.use_libcst,
            resolve_imports=opts.resolve_imports,
            max_edges_per_file=opts.max_edges_per_file,
        )
        edges = _collect_all_edges(paths, collection_ctx)
        log.info("call_graph: Collected %d edges from %d files", len(edges), len(paths))

        node_count = _persist_nodes(
            gateway, _build_nodes_from_goids(gateway, repo, commit), repo, commit
        )
        edge_count = _persist_edges(gateway, edges, repo, commit)

        log.info("call_graph: Persisted %d nodes, %d edges", node_count, edge_count)

        return CallGraphExtractResult(
            success=True,
            node_count=node_count,
            edge_count=edge_count,
            table_counts={
                CALL_GRAPH_NODES_TABLE_KEY: node_count,
                CALL_GRAPH_EDGES_TABLE_KEY: edge_count,
            },
        )

    except Exception as exc:
        log.exception("Call graph extraction failed")
        return CallGraphExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target=CALL_GRAPH_TARGET_NAME, node_type="materialize")
def t__call_graph(
    env: BuildEnv,
    graph: TargetGraph,
    call_graph__execution_result: ExecutionResult,
) -> TargetRunRecord:
    """Materialize call graph target with validation.

    This is the entry point for the call_graph target. It orchestrates
    call graph extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    call_graph__execution_result
        Execution result derived from upstream extract node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, CALL_GRAPH_TARGET_NAME, call_graph__execution_result)


__all__ = [
    "CallGraphExtractResult",
    "t__call_graph",
    "t__call_graph__extract",
]
