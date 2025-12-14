"""Symbol use graph builder plugin.

This module provides the symbol use edge builder as a build target plugin.

Architecture
------------
The symbol uses plugin performs the following steps:

1. Load symbol occurrences from SCIP index data
2. Build definition map from symbol occurrences
3. Build use edges connecting definitions to uses
4. Persist to graph.symbol_use_edges
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins._helpers import filter_mapping, is_test_path
from codeintel.build.plugins.graphs.builders.symbol_uses_options import SymbolUsesOptions
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.graphs.compute import symbols as symbols_compute
from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


SYMBOL_USES_METADATA = CorePluginMetadata(
    name="graphs.symbol_uses",
    version="3.0.0",
    description="Build symbol usage graph.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="edges",
    provides=("graph.symbol_uses",),
    requires=("core.scip_occurrences", "core.modules", "core.goids"),
    produces_tables=("graph.symbol_use_edges",),
    consumes_tables=("core.scip_occurrences", "core.modules", "core.goids"),
    supports_incremental=False,
    scope_aware=True,
    options_model=SymbolUsesOptions,
    extra={"graph_kinds": ("symbol_use",)},
)


def _filter_occurrences(
    occurrences: list[symbols_compute.SymbolOccurrence],
    options: SymbolUsesOptions,
) -> list[symbols_compute.SymbolOccurrence]:
    """Filter symbol occurrences by scope and test inclusion.

    Returns
    -------
    list[SymbolOccurrence]
        Filtered symbol occurrences.
    """
    scope_prefixes = tuple(options.scope_paths) if options.scope_paths else None
    filtered: list[symbols_compute.SymbolOccurrence] = []
    for occurrence in occurrences:
        path = occurrence.rel_path
        if scope_prefixes and not path.startswith(scope_prefixes):
            continue
        if not options.include_tests and is_test_path(path):
            continue
        filtered.append(occurrence)
    return filtered


def build_scip_candidates(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, tuple[str, ...]]:
    """Build SCIP candidate lookup from symbol occurrences.

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
    dict[str, tuple[str, ...]]
        Mapping of use path to tuple of potential definition paths.
    """
    try:
        occurrences_tbl = gateway.ibis.table("core.scip_occurrences")
        repo_filter = cast("Any", occurrences_tbl.repo == repo)
        commit_filter = cast("Any", occurrences_tbl.commit == commit)
        expr = (
            occurrences_tbl.filter(repo_filter & commit_filter)
            .select(occurrences_tbl.symbol, occurrences_tbl.rel_path, occurrences_tbl.roles)
            .distinct()
            .order_by(occurrences_tbl.symbol, occurrences_tbl.rel_path)
        )
        df = expr.execute()
        if getattr(df, "empty", True):
            return {}

        occurrences: list[symbols_compute.SymbolOccurrence] = []
        for symbol, rel_path, roles in df.itertuples(index=False, name=None):
            occurrences.append(
                symbols_compute.SymbolOccurrence(
                    symbol=str(symbol),
                    rel_path=normalize_rel_path(str(rel_path)),
                    line=0,
                    roles=symbols_compute.parse_symbol_roles(roles),
                )
            )

        def_map = symbols_compute.build_def_map(occurrences)

        return {
            use_path: tuple(sorted(def_paths))
            for use_path, def_paths in symbols_compute.build_use_def_mapping(
                occurrences, def_map
            ).items()
        }
    except DuckDBError:
        return {}


def _load_symbol_occurrences(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[symbols_compute.SymbolOccurrence]:
    """Load symbol occurrences from SCIP index.

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
    list[SymbolOccurrence]
        Symbol occurrences.
    """
    try:
        occurrences_tbl = gateway.ibis.table("core.scip_occurrences")
        repo_filter = cast("Any", occurrences_tbl.repo == repo)
        commit_filter = cast("Any", occurrences_tbl.commit == commit)
        expr = (
            occurrences_tbl.filter(repo_filter & commit_filter)
            .select(
                occurrences_tbl.symbol,
                occurrences_tbl.rel_path,
                occurrences_tbl.start_line,
                occurrences_tbl.roles,
            )
            .order_by(occurrences_tbl.symbol, occurrences_tbl.rel_path, occurrences_tbl.start_line)
        )
        rows = expr.execute()
        return [
            symbols_compute.SymbolOccurrence(
                symbol=str(symbol),
                rel_path=normalize_rel_path(str(rel_path)),
                line=int(start_line) if start_line is not None else 0,
                roles=symbols_compute.parse_symbol_roles(roles),
            )
            for symbol, rel_path, start_line, roles in rows.itertuples(index=False, name=None)
        ]
    except DuckDBError:
        return []


def _load_module_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module map from core.modules.

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
        Mapping of relative path to module name.
    """
    try:
        modules_tbl = gateway.ibis.table("core.modules")
        repo_filter = cast("Any", modules_tbl.repo == repo)
        commit_filter = cast("Any", modules_tbl.commit == commit)
        expr = modules_tbl.filter(repo_filter & commit_filter).select(
            modules_tbl.rel_path, modules_tbl.module_name
        )
        rows = expr.execute()
        return {
            normalize_rel_path(str(rel_path)): str(module_name)
            for rel_path, module_name in rows.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


def _load_path_to_goid_map(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, int]:
    """Load a mapping of file paths to their module/file GOID.

    This maps each file to a single representative GOID for that file.
    For function-level tracking, we use the first function GOID in each file.

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
        Mapping of relative path to representative GOID.
    """
    try:
        goids_tbl = gateway.ibis.table("core.goids")
        repo_filter = cast("Any", goids_tbl.repo == repo)
        commit_filter = cast("Any", goids_tbl.commit == commit)
        kind_filter = cast("Any", goids_tbl.kind == "function")
        expr = (
            goids_tbl.filter(repo_filter & commit_filter & kind_filter)
            .group_by(goids_tbl.rel_path)
            .aggregate(goid=goids_tbl.goid_h128.min())
        )
        rows = expr.execute()
        return {
            normalize_rel_path(str(rel_path)): int(goid)
            for rel_path, goid in rows.itertuples(index=False, name=None)
            if goid is not None
        }
    except DuckDBError:
        return {}


def _enrich_edges_with_goids(
    edges: list[symbols_compute.SymbolUseEdge],
    path_to_goid: dict[str, int],
) -> list[symbols_compute.SymbolUseEdge]:
    """Enrich symbol use edges with GOID information.

    Parameters
    ----------
    edges
        Symbol use edges without GOIDs.
    path_to_goid
        Mapping of file paths to representative GOIDs.

    Returns
    -------
    list[SymbolUseEdge]
        Edges enriched with def_goid and use_goid.
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


def _persist_symbol_use_edges(
    gateway: StorageGateway,
    edges: list[symbols_compute.SymbolUseEdge],
    repo: str,
    commit: str,
) -> int:
    """Persist symbol use edges.

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

    rows = symbols_compute.edges_to_rows(edges)

    gateway.policy.ensure_table("graph.symbol_use_edges")
    gateway.policy.delete_for_snapshot("graph.symbol_use_edges", repo=repo, commit=commit)
    gateway.policy.bulk_insert("graph.symbol_use_edges", [row.to_tuple() for row in rows])
    return len(rows)


class SymbolUsesPlugin(MetadataPlugin):
    """Build symbol usage graph.

    This plugin performs full symbol use analysis:
    1. Loads symbol occurrences from SCIP index
    2. Builds definition map from occurrences
    3. Creates use edges connecting definitions to uses
    4. Persists to graph.symbol_use_edges

    Outputs
    -------
    - graph.symbol_use_edges: Symbol use relationships
    """

    _core_metadata: ClassVar[CorePluginMetadata] = SYMBOL_USES_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute symbol use analysis.

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
        opts = self.resolve_options(SymbolUsesOptions)
        snapshot = ctx.snapshot

        gateway = ctx.gateway
        repo = snapshot.repo
        commit = snapshot.commit

        try:
            occurrences = _load_symbol_occurrences(gateway, repo, commit)
            occurrences = _filter_occurrences(occurrences, opts)

            if not occurrences:
                log.info("symbol_uses: No symbol occurrences found, skipping")
                return TargetResult.succeeded(row_counts={"graph.symbol_use_edges": 0})

            log.info("symbol_uses: Loaded %d symbol occurrences", len(occurrences))

            module_by_path = filter_mapping(
                _load_module_map(gateway, repo, commit),
                scope_paths=opts.scope_paths,
                include_tests=opts.include_tests,
            )
            log.debug("symbol_uses: Loaded %d module mappings", len(module_by_path))

            def_map = symbols_compute.build_def_map(occurrences)
            log.debug("symbol_uses: Built def map with %d entries", len(def_map))

            edges = symbols_compute.build_use_edges(occurrences, def_map, module_by_path)
            log.info("symbol_uses: Built %d use edges", len(edges))

            path_to_goid = filter_mapping(
                _load_path_to_goid_map(gateway, repo, commit),
                scope_paths=opts.scope_paths,
                include_tests=opts.include_tests,
            )
            log.debug("symbol_uses: Loaded %d path->goid mappings", len(path_to_goid))
            edges = _enrich_edges_with_goids(edges, path_to_goid)

            edge_count = _persist_symbol_use_edges(gateway, edges, repo, commit)
            log.info("symbol_uses: Persisted %d edges", edge_count)

            return TargetResult.succeeded(row_counts={"graph.symbol_use_edges": edge_count})
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Symbol use analysis failed: {e}")


__all__ = ["SymbolUsesPlugin", "build_scip_candidates"]
