"""Call graph builder plugin.

This module provides the call graph builder as a build target plugin.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import libcst as cst

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import CallGraphStepConfig
from codeintel.config.datasets import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    call_graph_node_to_tuple,
)
from codeintel.graphs.adapters.callgraph_persistence import (
    dedupe_edge_rows,
    persist_call_graph_edges,
)
from codeintel.graphs.catalog import (
    FunctionCatalogService,
)
from codeintel.graphs.compute.callgraph import (
    EdgeResolutionContext,
    collect_aliases,
    collect_edges_ast,
    collect_edges_cst,
)
from codeintel.graphs.plugins.builders import symbol_uses
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallGraphRunScope:
    """Identify the repository snapshot and filesystem root."""

    repo: str
    commit: str
    repo_root: Path


@dataclass(frozen=True)
class CallGraphInputs:
    """Resolution inputs and optional collectors for call graph edges."""

    global_callee_by_name: dict[str, int]
    scip_candidates_by_use: dict[str, tuple[str, ...]]
    def_goids_by_path: dict[str, int]
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None


def _log_repo_state(gateway: StorageGateway, repo: str, commit: str) -> None:
    """Log current module/GOID counts to aid validation diagnostics."""
    con = gateway.con
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
    rows = con.execute(
        """
        SELECT qualname, function_goid_h128
        FROM core.goids
        WHERE repo = ? AND commit = ? AND kind = 'function'
        """,
        [repo, commit],
    ).fetchall()
    return {row[0]: int(row[1]) for row in rows}


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
    rows = con.execute(
        """
        SELECT rel_path, function_goid_h128
        FROM core.goids
        WHERE repo = ? AND commit = ? AND kind = 'module'
        """,
        [repo, commit],
    ).fetchall()
    return {row[0]: int(row[1]) for row in rows}


def _collect_call_edges(
    catalog: object,
    cfg: CallGraphStepConfig,
    inputs: CallGraphInputs,
) -> list[CallGraphEdgeRow]:
    """Collect call graph edges by parsing source files.

    Parameters
    ----------
    catalog
        Function catalog with function metadata.
    cfg
        Build configuration.
    inputs
        Resolution inputs.

    Returns
    -------
    list[CallGraphEdgeRow]
        Collected call graph edges.
    """
    edges: list[CallGraphEdgeRow] = []
    resolution_ctx = EdgeResolutionContext(
        global_callee_by_name=inputs.global_callee_by_name,
        scip_candidates_by_use=inputs.scip_candidates_by_use,
        def_goids_by_path=inputs.def_goids_by_path,
    )

    for goid, meta in catalog.function_by_goid.items():  # type: ignore[attr-defined]
        if not (meta.rel_path and meta.span):
            continue
        file_path = cfg.repo_root / meta.rel_path
        if not file_path.exists():
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
            aliases = collect_aliases(source, meta.rel_path)
            if inputs.cst_collector:
                edges.extend(
                    inputs.cst_collector(
                        source,
                        meta,
                        goid,
                        cfg.repo,
                        cfg.commit,
                        resolution_ctx,
                        aliases,
                    )
                )
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError) as e:
            log.warning("Failed to parse %s: %s", file_path, e)

    return edges


class CallGraphPlugin(TargetPlugin):
    """Build call graph nodes and edges.

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
            Execution result.
        """
        # Build config
        cfg = CallGraphStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        gateway = ctx.gateway
        repo = ctx.repo
        commit = ctx.commit

        _log_repo_state(gateway, repo, commit)

        # Build lookups
        global_callee_by_name = _build_global_callee_lookup(gateway, repo, commit)
        def_goids_by_path = _build_def_goids_by_path(gateway, repo, commit)

        # Get SCIP candidates from symbol_uses if available
        scip_candidates_by_use: dict[str, tuple[str, ...]] = {}
        if ctx.resources.catalog is not None:
            scip_candidates_by_use = symbol_uses.build_scip_candidates(
                gateway, repo, commit
            )

        inputs = CallGraphInputs(
            global_callee_by_name=global_callee_by_name,
            scip_candidates_by_use=scip_candidates_by_use,
            def_goids_by_path=def_goids_by_path,
            cst_collector=collect_edges_cst,
            ast_collector=collect_edges_ast,
        )

        try:
            row_counts = self._build_call_graph(
                gateway,
                cfg,
                inputs,
            )
            return TargetResult.succeeded(row_counts=row_counts)
        except (RuntimeError, ValueError, OSError, DuckDBError) as e:
            return TargetResult.failed(f"Call graph build failed: {e}")

    def _build_call_graph(
        self,
        gateway: StorageGateway,
        cfg: CallGraphStepConfig,
        inputs: CallGraphInputs,
    ) -> dict[str, int]:
        """Build call graph nodes and edges.

        Parameters
        ----------
        gateway
            Storage gateway.
        cfg
            Configuration.
        inputs
            Resolution inputs.

        Returns
        -------
        dict[str, int]
            Row counts.
        """
        _ = self  # Instance method for potential future extension

        # Get catalog service
        storage = IngestStorageService(gateway)
        catalog = FunctionCatalogService(storage, cfg.repo, cfg.commit).catalog()

        # Build nodes from catalog
        nodes: list[CallGraphNodeRow] = []
        now = datetime.now(tz=UTC)

        for goid, meta in catalog.function_by_goid.items():
            node = CallGraphNodeRow(
                function_goid_h128=goid,
                repo=cfg.repo,
                commit=cfg.commit,
                qualname=meta.qualname,
                rel_path=meta.rel_path,
                language="python",
                kind="function",
                start_line=meta.span.start_line if meta.span else 0,
                end_line=meta.span.end_line if meta.span else 0,
                created_at=now,
            )
            nodes.append(node)

        # Persist nodes
        gateway.con.execute(
            "DELETE FROM graphs.call_graph_nodes WHERE repo = ? AND commit = ?",
            [cfg.repo, cfg.commit],
        )
        if nodes:
            gateway.con.executemany(
                """
                INSERT INTO graphs.call_graph_nodes
                (function_goid_h128, repo, commit, qualname, rel_path,
                 language, kind, start_line, end_line, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [call_graph_node_to_tuple(n) for n in nodes],
            )

        # Collect edges
        edges = _collect_call_edges(catalog, cfg, inputs)

        # Dedupe and persist edges
        edges = dedupe_edge_rows(edges)
        persist_call_graph_edges(gateway, edges, cfg.repo, cfg.commit)

        return {
            "graphs.call_graph_nodes": len(nodes),
            "graphs.call_graph_edges": len(edges),
        }


__all__ = ["CallGraphPlugin"]
