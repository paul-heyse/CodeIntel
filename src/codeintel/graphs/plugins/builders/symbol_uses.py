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
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import SymbolUsesStepConfig
from codeintel.graphs.compute import symbols as symbols_compute
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.ingestion.infrastructure.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


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
    con = gateway.con
    try:
        # Load occurrences from core.scip_occurrences
        rows = con.execute(
            """
            SELECT DISTINCT
                o.symbol,
                o.rel_path,
                o.roles
            FROM core.scip_occurrences o
            WHERE o.repo = ? AND o.commit = ?
            ORDER BY o.symbol, o.rel_path
            """,
            [repo, commit],
        ).fetchall()

        if not rows:
            return {}

        # Parse occurrences
        occurrences: list[symbols_compute.SymbolOccurrence] = []
        for symbol, rel_path, roles in rows:
            occurrences.append(
                symbols_compute.SymbolOccurrence(
                    symbol=str(symbol),
                    rel_path=normalize_rel_path(str(rel_path)),
                    line=0,  # We don't need exact line for this use case
                    roles=symbols_compute.parse_symbol_roles(roles),
                )
            )

        # Build def map
        def_map = symbols_compute.build_def_map(occurrences)

        # Build use -> def paths mapping
        return {
            use_path: tuple(sorted(def_paths))
            for use_path, def_paths in symbols_compute.build_use_def_mapping(
                occurrences, def_map
            ).items()
        }
    except Exception:  # noqa: BLE001
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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT
                symbol,
                rel_path,
                start_line,
                roles
            FROM core.scip_occurrences
            WHERE repo = ? AND commit = ?
            ORDER BY symbol, rel_path, start_line
            """,
            [repo, commit],
        ).fetchall()

        return [
            symbols_compute.SymbolOccurrence(
                symbol=str(row[0]),
                rel_path=normalize_rel_path(str(row[1])),
                line=int(row[2]) if row[2] is not None else 0,
                roles=symbols_compute.parse_symbol_roles(row[3]),
            )
            for row in rows
        ]
    except Exception:  # noqa: BLE001
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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT rel_path, module_name
            FROM core.modules
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
        return {normalize_rel_path(str(row[0])): str(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT rel_path, MIN(goid_h128) as goid
            FROM core.goids
            WHERE repo = ? AND commit = ? AND kind = 'function'
            GROUP BY rel_path
            """,
            [repo, commit],
        ).fetchall()
        return {normalize_rel_path(str(row[0])): int(row[1]) for row in rows if row[1] is not None}
    except Exception:  # noqa: BLE001
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

    # Convert SymbolUseEdge to SymbolUseRow using edges_to_rows
    rows = symbols_compute.edges_to_rows(edges)

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graph.symbol_use_edges",
        [row.to_tuple() for row in rows],
        delete_params=[repo, commit],
        scope="symbol_use_edges",
    )
    return len(rows)


class SymbolUsesPlugin(TargetPlugin):
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

    plugin_name: ClassVar[str] = "symbol_uses"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build symbol usage graph."

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
        _ = self  # Protocol method requires instance

        # Build config - SymbolUsesStepConfig requires paths
        cfg = SymbolUsesStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
        )

        gateway = ctx.gateway
        repo = cfg.repo
        commit = cfg.commit

        try:
            # Step 1: Load symbol occurrences
            occurrences = _load_symbol_occurrences(gateway, repo, commit)

            if not occurrences:
                log.info("symbol_uses: No symbol occurrences found, skipping")
                return TargetResult.succeeded(row_counts={"graph.symbol_use_edges": 0})

            log.info("symbol_uses: Loaded %d symbol occurrences", len(occurrences))

            # Step 2: Load module map
            module_by_path = _load_module_map(gateway, repo, commit)
            log.debug("symbol_uses: Loaded %d module mappings", len(module_by_path))

            # Step 3: Build definition map
            def_map = symbols_compute.build_def_map(occurrences)
            log.debug("symbol_uses: Built def map with %d entries", len(def_map))

            # Step 4: Build use edges
            edges = symbols_compute.build_use_edges(occurrences, def_map, module_by_path)
            log.info("symbol_uses: Built %d use edges", len(edges))

            # Step 5: Enrich edges with GOIDs for function-level tracking
            path_to_goid = _load_path_to_goid_map(gateway, repo, commit)
            log.debug("symbol_uses: Loaded %d path->goid mappings", len(path_to_goid))
            edges = _enrich_edges_with_goids(edges, path_to_goid)

            # Step 6: Persist
            edge_count = _persist_symbol_use_edges(gateway, edges, repo, commit)
            log.info("symbol_uses: Persisted %d edges", edge_count)

            return TargetResult.succeeded(row_counts={"graph.symbol_use_edges": edge_count})
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Symbol use analysis failed: {e}")


__all__ = ["SymbolUsesPlugin", "build_scip_candidates"]
