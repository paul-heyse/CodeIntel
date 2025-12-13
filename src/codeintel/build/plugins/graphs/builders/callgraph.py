"""Call graph builder plugin.

This module provides the call graph builder as a build target plugin.

Architecture
------------
The call graph plugin performs the following steps:

1. Load function spans from `core.goids` to build FunctionSpanIndex
2. Build global callee map (qualname -> GOID) for resolution
3. Build module GOID map (path -> module GOID) for SCIP fallback
4. For each Python file with functions:
   - Build local callee map from function index
   - Collect import aliases from the file
   - Create EdgeResolutionContext with all lookup maps
   - Parse file and collect call edges via LibCST (or AST fallback)
5. Persist deduplicated edges to graph.call_graph_edges
6. Persist nodes to graph.call_graph_nodes
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import ibis
import libcst as cst

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins._metadata import to_plugin_metadata
from codeintel.build.plugins.graphs.builders.callgraph_options import CallGraphOptions
from codeintel.config.datasets import (
    CallGraphNodeRow,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.graphs.catalog import (
    load_function_index,
)
from codeintel.graphs.compute.callgraph import (
    EdgeResolutionContext,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
)
from codeintel.graphs.compute.callgraph.persistence import dedupe_edge_rows
from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.config.datasets import (
        CallGraphEdgeRow,
    )
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.graphs.catalog import (
        FunctionSpanIndex,
    )
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


CALLGRAPH_METADATA = CorePluginMetadata(
    name="graphs.callgraph",
    version="3.0.0",
    description="Build call graph nodes and edges.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="edges",
    provides=("graph.callgraph",),
    requires=("core.goids",),
    produces_tables=(
        "graph.call_graph_nodes",
        "graph.call_graph_edges",
    ),
    consumes_tables=(
        "core.goids",
        "core.modules",
    ),
    supports_incremental=False,
    scope_aware=True,
    options_model=CallGraphOptions,
    resource_hints={
        "max_memory_mb": 1024,
    },
    extra={"graph_kinds": ("callgraph",)},
)


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
            "call_graph_builder repo_state modules=%d goids=%d (module_kind=%d)",
            module_count,
            goid_count,
            module_goid_count,
        )
    except DuckDBError as exc:
        log.debug("call_graph_builder: Could not query repo state: %s", exc)


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
        repo_filter = cast("Any", goids_tbl.repo == repo)
        commit_filter = cast("Any", goids_tbl.commit == commit)
        kind_filter = cast("Any", goids_tbl.kind.isin(cast("Any", ["function", "method"])))
        expr = (
            goids_tbl.filter(repo_filter & commit_filter & kind_filter)
            .select(goids_tbl.qualname, goids_tbl.goid_h128)
            .order_by(goids_tbl.qualname)
        )
        rows = expr.execute()
        return {
            str(qualname): int(goid) for qualname, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


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
        repo_filter = cast("Any", goids_tbl.repo == repo)
        commit_filter = cast("Any", goids_tbl.commit == commit)
        kind_filter = cast("Any", goids_tbl.kind == "module")
        expr = (
            goids_tbl.filter(repo_filter & commit_filter & kind_filter)
            .select(goids_tbl.rel_path, goids_tbl.goid_h128)
            .order_by(goids_tbl.rel_path)
        )
        rows = expr.execute()
        return {
            normalize_rel_path(str(rel_path)): int(goid)
            for rel_path, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


def _get_source_root(gateway: StorageGateway, repo: str, commit: str) -> Path | None:
    """Retrieve source root from core.snapshots.

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
    Path | None
        Absolute path to source root, or None if not found.
    """
    try:
        snapshots = gateway.ibis.table("core.snapshots")
        repo_filter = cast("Any", snapshots.repo == repo)
        commit_filter = cast("Any", snapshots.commit == commit)
        expr = snapshots.filter(repo_filter & commit_filter).select(snapshots.source_root).limit(1)
        rows = expr.execute()
        if getattr(rows, "empty", True):
            return None
        source_root = rows.iloc[0][0]
        if source_root:
            return Path(str(source_root))
    except DuckDBError as exc:
        log.debug("callgraph: Could not get source root: %s", exc)
    return None


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
        repo_filter = cast("Any", goids_tbl.repo == repo)
        commit_filter = cast("Any", goids_tbl.commit == commit)
        kind_filter = cast("Any", goids_tbl.kind.isin(cast("Any", ["function", "method"])))

        language_expr = (
            goids_tbl.language if "language" in goids_tbl.columns else ibis.literal("python")
        )
        rel_path_expr = goids_tbl.rel_path if "rel_path" in goids_tbl.columns else ibis.literal("")
        kind_expr = goids_tbl.kind if "kind" in goids_tbl.columns else ibis.literal("function")

        expr = (
            goids_tbl.filter(repo_filter & commit_filter & kind_filter)
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
        log.debug("callgraph: Could not build nodes from GOIDs: %s", exc)
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

    gateway.policy.ensure_table("graph.call_graph_nodes")
    gateway.policy.delete_for_snapshot("graph.call_graph_nodes", repo=repo, commit=commit)
    gateway.policy.bulk_insert(
        "graph.call_graph_nodes",
        [call_graph_node_to_tuple(node) for node in nodes],
    )
    return len(nodes)


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

    gateway.policy.ensure_table("graph.call_graph_edges")
    gateway.policy.delete_for_snapshot("graph.call_graph_edges", repo=repo, commit=commit)
    gateway.policy.bulk_insert(
        "graph.call_graph_edges",
        [call_graph_edge_to_tuple(e) for e in serialized],
    )
    return len(serialized)


@dataclass(frozen=True)
class _EdgeCollectionContext:
    """Context for edge collection across files."""

    function_index: FunctionSpanIndex
    global_callees: dict[str, int]
    def_goids_by_path: dict[str, int]
    source_root: Path
    repo: str
    commit: str
    use_libcst: bool
    resolve_imports: bool
    max_edges_per_file: int


def _collect_all_edges(
    paths: list[str],
    ctx: _EdgeCollectionContext,
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


def _filter_paths_by_scope(paths: list[str], scope_paths: list[str] | None) -> list[str]:
    """Filter paths by configured scope prefixes.

    Returns
    -------
    list[str]
        Filtered list of relative paths.
    """
    if not scope_paths:
        return paths
    prefixes = tuple(scope_paths)
    return [path for path in paths if path.startswith(prefixes)]


class CallGraphPlugin(TargetPlugin):
    """Build call graph nodes and edges.

    This plugin performs full call graph construction:
    1. Loads function metadata from core.goids
    2. Parses source files to collect call edges
    3. Resolves callees using local/global/import maps
    4. Persists nodes and edges to graph.call_graph_*

    Outputs
    -------
    - graph.call_graph_nodes: Call graph nodes
    - graph.call_graph_edges: Call graph edges
    """

    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build call graph nodes and edges."
    _core_metadata: ClassVar[CorePluginMetadata] = CALLGRAPH_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Protocol-compatible metadata.
        """
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata.

        Returns
        -------
        CorePluginMetadata
            Canonical metadata definition.
        """
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> CallGraphOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        CallGraphOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return CallGraphOptions(**dynamic_overrides)
            return CallGraphOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            CallGraphOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute call graph construction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result with row counts.
        """
        _ = self
        opts = self.resolve_options()
        snapshot = ctx.snapshot
        gateway, repo, commit = ctx.gateway, snapshot.repo, snapshot.commit

        try:
            _log_repo_state(gateway, repo, commit)

            function_index = load_function_index(gateway, repo=repo, commit=commit)
            paths = _filter_paths_by_scope(function_index.paths(), opts.scope_paths)

            if not paths:
                log.info("callgraph: No functions found, skipping")
                return TargetResult.succeeded(
                    row_counts={"graph.call_graph_nodes": 0, "graph.call_graph_edges": 0}
                )

            global_callees = _build_global_callee_lookup(gateway, repo, commit)
            def_goids = _build_def_goids_by_path(gateway, repo, commit)

            source_root = (
                snapshot.repo_root or _get_source_root(gateway, repo, commit) or Path.cwd()
            )

            collection_ctx = _EdgeCollectionContext(
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
            log.info("callgraph: Collected %d edges from %d files", len(edges), len(paths))

            node_count = _persist_nodes(
                gateway, _build_nodes_from_goids(gateway, repo, commit), repo, commit
            )
            edge_count = _persist_edges(gateway, edges, repo, commit)

            log.info("callgraph: Persisted %d nodes, %d edges", node_count, edge_count)
            return TargetResult.succeeded(
                row_counts={
                    "graph.call_graph_nodes": node_count,
                    "graph.call_graph_edges": edge_count,
                }
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Call graph build failed: {e}")


__all__ = [
    "CALLGRAPH_METADATA",
    "CallGraphPlugin",
]
