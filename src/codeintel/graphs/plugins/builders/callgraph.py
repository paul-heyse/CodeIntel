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
5. Persist deduplicated edges to graphs.call_graph_edges
6. Persist nodes to graphs.call_graph_nodes
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import libcst as cst

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import CallGraphStepConfig
from codeintel.config.datasets import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    call_graph_edge_to_tuple,
    call_graph_node_to_tuple,
)
from codeintel.graphs.adapters.callgraph_persistence import dedupe_edge_rows
from codeintel.graphs.catalog import (
    FunctionSpanIndex,
    load_function_index,
)
from codeintel.graphs.compute.callgraph import (
    EdgeResolutionContext,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
)
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.ingestion.infrastructure.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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
    con = gateway.con
    try:
        modules = con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        goids = con.execute(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        module_goids = con.execute(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind = 'module'",
            [repo, commit],
        ).fetchone()
        log.info(
            "call_graph_builder repo_state modules=%d goids=%d (module_kind=%d)",
            modules[0] if modules else 0,
            goids[0] if goids else 0,
            module_goids[0] if module_goids else 0,
        )
    except Exception:  # noqa: BLE001
        # Tables may not exist yet in early pipeline stages
        log.debug("call_graph_builder: Could not query repo state")


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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT qualname, goid_h128
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
            """,
            [repo, commit],
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT rel_path, goid_h128
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind = 'module'
            """,
            [repo, commit],
        ).fetchall()
        return {normalize_rel_path(str(row[0])): int(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
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
    con = gateway.con
    try:
        row = con.execute(
            "SELECT source_root FROM core.snapshots WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        if row and row[0]:
            return Path(row[0])
    except Exception as e:  # noqa: BLE001
        log.debug("callgraph: Could not get source root: %s", e)
    return None


def _collect_edges_for_file(
    rel_path: str,
    file_path: Path,
    context: EdgeResolutionContext,
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

    # Try LibCST first (more accurate positions)
    try:
        module = cst.parse_module(source)
        return collect_edges_cst(rel_path, module, context)
    except cst.ParserSyntaxError:
        # Fall back to AST
        return collect_edges_ast(rel_path, file_path, context)


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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT
                goid_h128,
                'python' AS language,
                kind,
                COALESCE(arity, 0) AS arity,
                is_public,
                rel_path
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
            """,
            [repo, commit],
        ).fetchall()
    except Exception as e:  # noqa: BLE001
        log.debug("callgraph: Could not build nodes from GOIDs: %s", e)
        return []

    return [
        CallGraphNodeRow(
            goid_h128=int(goid_h128),
            language=str(language),
            kind=str(kind),
            arity=int(arity) if arity is not None else 0,
            is_public=bool(is_public) if is_public is not None else True,
            rel_path=str(rel_path),
        )
        for goid_h128, language, kind, arity, is_public, rel_path in rows
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

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graphs.call_graph_nodes",
        [call_graph_node_to_tuple(node) for node in nodes],
        delete_params=[repo, commit],
        scope="call_graph_nodes",
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

    # Deduplicate edges
    unique_edges = dedupe_edge_rows(edges)

    # Serialize evidence_json fields
    serialized: list[CallGraphEdgeRow] = []
    for edge in unique_edges:
        evidence = edge["evidence_json"]
        if isinstance(evidence, dict):
            serialized.append({**edge, "evidence_json": json.dumps(evidence)})
        else:
            serialized.append(edge)

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graphs.call_graph_edges",
        [call_graph_edge_to_tuple(e) for e in serialized],
        delete_params=[repo, commit],
        scope="call_graph_edges",
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
        if file_path.exists():
            with contextlib.suppress(OSError, UnicodeDecodeError, cst.ParserSyntaxError):
                import_aliases = collect_aliases(cst.parse_module(file_path.read_text(encoding="utf8")))

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
        all_edges.extend(_collect_edges_for_file(rel_path, file_path, context))

    return all_edges


class CallGraphPlugin(TargetPlugin):
    """Build call graph nodes and edges.

    This plugin performs full call graph construction:
    1. Loads function metadata from core.goids
    2. Parses source files to collect call edges
    3. Resolves callees using local/global/import maps
    4. Persists nodes and edges to graphs.call_graph_*

    Outputs
    -------
    - graphs.call_graph_nodes: Call graph nodes
    - graphs.call_graph_edges: Call graph edges
    """

    plugin_name: ClassVar[str] = "callgraph"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build call graph nodes and edges."

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
        _ = self  # Protocol method requires instance
        cfg = CallGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, cfg.repo, cfg.commit

        try:
            _log_repo_state(gateway, repo, commit)

            # Load function index and get paths
            function_index = load_function_index(gateway, repo=repo, commit=commit)
            paths = function_index.paths()

            if not paths:
                log.info("callgraph: No functions found, skipping")
                return TargetResult.succeeded(
                    row_counts={"graphs.call_graph_nodes": 0, "graphs.call_graph_edges": 0}
                )

            # Build lookup maps
            global_callees = _build_global_callee_lookup(gateway, repo, commit)
            def_goids = _build_def_goids_by_path(gateway, repo, commit)
            source_root = _get_source_root(gateway, repo, commit) or Path.cwd()

            # Collect and persist edges
            collection_ctx = _EdgeCollectionContext(
                function_index=function_index,
                global_callees=global_callees,
                def_goids_by_path=def_goids,
                source_root=source_root,
                repo=repo,
                commit=commit,
            )
            edges = _collect_all_edges(paths, collection_ctx)
            log.info("callgraph: Collected %d edges from %d files", len(edges), len(paths))

            # Build and persist nodes
            node_count = _persist_nodes(
                gateway, _build_nodes_from_goids(gateway, repo, commit), repo, commit
            )
            edge_count = _persist_edges(gateway, edges, repo, commit)

            log.info("callgraph: Persisted %d nodes, %d edges", node_count, edge_count)
            return TargetResult.succeeded(
                row_counts={"graphs.call_graph_nodes": node_count, "graphs.call_graph_edges": edge_count}
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Call graph build failed: {e}")


__all__ = ["CallGraphPlugin"]
