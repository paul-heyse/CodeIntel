"""Native Hamilton implementation for symbol_uses target.

This module implements symbol use graph construction as a native Hamilton pipeline with:
- t__symbol_uses__extract: Load SCIP occurrences and build use edges
- t__symbol_uses: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import filter_mapping, persist_rows
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.options.graphs import SymbolUsesOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import symbols as symbols_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class SymbolUsesExtractResult:
    """Result from symbol uses extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    edge_count
        Number of symbol use edges extracted.
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
def _load_symbol_occurrences(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[symbols_compute.SymbolOccurrence]:
    """Load SCIP symbol occurrences from database.

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
    list[symbols_compute.SymbolOccurrence]
        Symbol occurrences for processing.
    """
    try:
        scip_tbl = gateway.ibis.table("core.scip_occurrences")
        expr = scip_tbl.filter(
            cast("Any", scip_tbl.repo == repo) & cast("Any", scip_tbl.commit == commit)
        ).select(scip_tbl.symbol, scip_tbl.rel_path, scip_tbl.line, scip_tbl.roles)
        rows = expr.execute()

        return [
            symbols_compute.SymbolOccurrence(
                symbol=str(symbol),
                rel_path=normalize_path(str(rel_path)),
                line=int(line or 0),
                roles=symbols_compute.parse_symbol_roles(roles),
            )
            for symbol, rel_path, line, roles in rows.itertuples(index=False, name=None)
        ]
    except DuckDBError:
        return []


@tag(node_type="helper")
def _load_module_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module name by path mapping.

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
    dict[str, str]
        Path to module name mapping.
    """
    try:
        modules_tbl = gateway.ibis.table("core.modules")
        expr = modules_tbl.filter(
            cast("Any", modules_tbl.repo == repo) & cast("Any", modules_tbl.commit == commit)
        ).select(modules_tbl.path, modules_tbl.module)
        rows = expr.execute()
        return {
            normalize_path(str(path)): str(module)
            for path, module in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag(node_type="helper")
def _load_path_to_goid_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Load GOID by path mapping.

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
        Path to module GOID mapping.
    """
    try:
        goids_tbl = gateway.ibis.table("core.goids")
        expr = goids_tbl.filter(
            cast("Any", goids_tbl.repo == repo)
            & cast("Any", goids_tbl.commit == commit)
            & cast("Any", goids_tbl.kind == "module")
        ).select(goids_tbl.rel_path, goids_tbl.goid_h128)
        rows = expr.execute()
        return {
            normalize_path(str(rel_path)): int(goid)
            for rel_path, goid in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag(node_type="helper")
def _enrich_edges_with_goids(
    edges: list[symbols_compute.SymbolUseEdge],
    path_to_goid: dict[str, int],
) -> list[symbols_compute.SymbolUseEdge]:
    """Enrich symbol use edges with GOIDs.

    Parameters
    ----------
    edges
        Symbol use edges to enrich.
    path_to_goid
        Path to GOID mapping.

    Returns
    -------
    list[symbols_compute.SymbolUseEdge]
        Edges enriched with GOIDs.
    """
    enriched: list[symbols_compute.SymbolUseEdge] = []
    for edge in edges:
        def_goid = path_to_goid.get(edge.def_path)
        use_goid = path_to_goid.get(edge.use_path)
        enriched.append(
            symbols_compute.SymbolUseEdge(
                symbol=edge.symbol,
                def_path=edge.def_path,
                use_path=edge.use_path,
                same_file=edge.same_file,
                same_module=edge.same_module,
                def_goid=def_goid,
                use_goid=use_goid,
            )
        )
    return enriched


@tag(domain="graphs", target="symbol_uses", node_type="compute")
def t__symbol_uses__extract(
    env: BuildEnv,
    t__scip: TargetRunRecord,
    t__modules: TargetRunRecord,
    t__goids: TargetRunRecord,
) -> SymbolUsesExtractResult:
    """Execute symbol use extraction from SCIP data.

    This is the compute node for the symbol_uses target. It loads SCIP
    symbol occurrences, builds definition maps, and creates use edges.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__scip
        Upstream SCIP target result (for dependency).
    t__modules
        Upstream modules target result (for dependency).
    t__goids
        Upstream GOIDs target result (for dependency).

    Returns
    -------
    SymbolUsesExtractResult
        Result containing edge count.

    Notes
    -----
    Produces:
    - graph.symbol_use_edges: Symbol use relationship edges
    """
    for name, record in [("scip", t__scip), ("modules", t__modules), ("goids", t__goids)]:
        if record.status != "succeeded":
            return SymbolUsesExtractResult(
                success=False,
                error=f"Upstream {name} target failed: {record.error}",
            )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = SymbolUsesOptions()

        occurrences = _load_symbol_occurrences(gateway, repo, commit)

        if not occurrences:
            log.info("symbol_uses: No SCIP occurrences found, skipping")
            return SymbolUsesExtractResult(
                success=True,
                edge_count=0,
                table_counts={"graph.symbol_use_edges": 0},
            )

        module_by_path = filter_mapping(
            _load_module_map(gateway, repo, commit),
            scope_paths=opts.scope_paths,
        )
        path_to_goid = _load_path_to_goid_map(gateway, repo, commit)

        def_map = symbols_compute.build_def_map(occurrences)
        edges = symbols_compute.build_use_edges(occurrences, def_map, module_by_path)

        enriched_edges = _enrich_edges_with_goids(edges, path_to_goid)
        rows = symbols_compute.edges_to_rows(enriched_edges)

        log.info("symbol_uses: Built %d edges from %d occurrences", len(rows), len(occurrences))

        edge_count = persist_rows(
            gateway,
            "graph.symbol_use_edges",
            rows,
            repo=repo,
            commit=commit,
        )

        log.info("symbol_uses: Persisted %d edges", edge_count)

        return SymbolUsesExtractResult(
            success=True,
            edge_count=edge_count,
            table_counts={"graph.symbol_use_edges": edge_count},
        )

    except Exception as exc:
        log.exception("Symbol uses extraction failed")
        return SymbolUsesExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="symbol_uses", node_type="materialize")
def t__symbol_uses(
    env: BuildEnv,
    graph: TargetGraph,
    t__symbol_uses__extract: SymbolUsesExtractResult,
) -> TargetRunRecord:
    """Materialize symbol uses target with validation.

    This is the entry point for the symbol_uses target. It orchestrates
    symbol use extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__symbol_uses__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "symbol_uses")

    if executor.should_skip():
        return executor.skip()

    if not t__symbol_uses__extract.success:
        return executor.fail(
            RuntimeError(t__symbol_uses__extract.error or "Symbol uses extraction failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__symbol_uses__extract.table_counts)

    return executor.execute(compute)


__all__ = [
    "SymbolUsesExtractResult",
    "t__symbol_uses",
    "t__symbol_uses__extract",
]
