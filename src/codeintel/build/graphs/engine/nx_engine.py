"""NetworkX-backed GraphEngine implementation.

This module provides the primary GraphEngine implementation using
NetworkX for graph representation and Parquet-backed datasets for
data loading. It is a hybrid service layer (not a Hamilton DAG
module) and avoids view-registry fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.build.graphs.engine import views
from codeintel.build.graphs.engine.cache import GraphCache, GraphCacheMetadata
from codeintel.build.graphs.engine.protocol import GraphKind
from codeintel.core.datasets.parquet_metadata import metadata_from_schema
from codeintel.core.datasets.paths import SnapshotIdError, dataset_snapshot_dir
from codeintel.core.hashing.fingerprint import stable_hash

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.build.graphs.engine.backend import BackendEnablement
    from codeintel.config.primitives import SnapshotRef

log = logging.getLogger(__name__)

_GRAPH_CACHE_TABLES: dict[GraphKind, tuple[str, ...]] = {
    GraphKind.CALL_GRAPH: ("graph.call_graph_edges", "graph.call_graph_nodes"),
    GraphKind.IMPORT_GRAPH: ("graph.import_graph_edges", "graph.import_modules"),
    GraphKind.SYMBOL_MODULE_GRAPH: ("graph.symbol_use_edges", "core.modules"),
    GraphKind.SYMBOL_FUNCTION_GRAPH: ("graph.symbol_use_edges",),
    GraphKind.CONFIG_MODULE_BIPARTITE: ("core.modules", "analytics.config_values"),
}


@dataclass
class NxGraphEngine:
    """NetworkX-backed GraphEngine powered by Parquet-backed datasets."""

    dataset_root_dir: Path | None
    snapshot: SnapshotRef
    use_gpu: bool = False
    effective_use_gpu: bool = False
    backend_info: BackendEnablement | None = None
    _cache: GraphCache = field(default_factory=GraphCache)

    def seed(self, kind: GraphKind, graph: nx.Graph | None) -> None:
        """
        Pre-populate the cache when a graph is already available.

        Parameters
        ----------
        kind : GraphKind
            Type of graph being cached.
        graph : nx.Graph | None
            Graph instance to cache, or None to skip.
        """
        self._cache.seed(kind, graph)

    @property
    def repo(self) -> str:
        """Repository identifier for the bound snapshot."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the bound snapshot."""
        return self.snapshot.commit

    def call_graph(self) -> nx.DiGraph:
        """
        Return the call graph for the configured snapshot.

        Returns
        -------
        nx.DiGraph
            Cached or freshly materialized call graph.
        """
        metadata = self._graph_cache_metadata(GraphKind.CALL_GRAPH)
        graph = self._cache.get(
            GraphKind.CALL_GRAPH,
            lambda: views.load_call_graph(
                self.dataset_root_dir,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
            metadata=metadata,
        )
        return cast("nx.DiGraph", graph)

    def load_call_graph(self) -> nx.DiGraph:
        """
        Alias for call_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.DiGraph
            Directed call graph.
        """
        return self.call_graph()

    def import_graph(self) -> nx.DiGraph:
        """
        Return the import graph for the configured snapshot.

        Returns
        -------
        nx.DiGraph
            Cached or freshly materialized import graph.
        """
        metadata = self._graph_cache_metadata(GraphKind.IMPORT_GRAPH)
        graph = self._cache.get(
            GraphKind.IMPORT_GRAPH,
            lambda: views.load_import_graph(
                self.dataset_root_dir,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
            metadata=metadata,
        )
        return cast("nx.DiGraph", graph)

    def load_import_graph(self) -> nx.DiGraph:
        """
        Alias for import_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.DiGraph
            Directed import graph.
        """
        return self.import_graph()

    def symbol_module_graph(self) -> nx.Graph:
        """
        Return the symbol coupling graph aggregated at module granularity.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized symbol-module graph.
        """
        metadata = self._graph_cache_metadata(GraphKind.SYMBOL_MODULE_GRAPH)
        return self._cache.get(
            GraphKind.SYMBOL_MODULE_GRAPH,
            lambda: views.load_symbol_module_graph(
                self.dataset_root_dir,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
            metadata=metadata,
        )

    def load_symbol_module_graph(self) -> nx.Graph:
        """
        Alias for symbol_module_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Symbol-module coupling graph.
        """
        return self.symbol_module_graph()

    def symbol_function_graph(self) -> nx.Graph:
        """
        Return the symbol coupling graph aggregated at function granularity.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized symbol-function graph.
        """
        metadata = self._graph_cache_metadata(GraphKind.SYMBOL_FUNCTION_GRAPH)
        return self._cache.get(
            GraphKind.SYMBOL_FUNCTION_GRAPH,
            lambda: views.load_symbol_function_graph(
                self.dataset_root_dir,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
            metadata=metadata,
        )

    def load_symbol_function_graph(self) -> nx.Graph:
        """
        Alias for symbol_function_graph to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Symbol-function coupling graph.
        """
        return self.symbol_function_graph()

    def config_module_bipartite(self) -> nx.Graph:
        """
        Return the config key <-> module bipartite graph.

        Returns
        -------
        nx.Graph
            Cached or freshly materialized config bipartite graph.
        """
        metadata = self._graph_cache_metadata(GraphKind.CONFIG_MODULE_BIPARTITE)
        return self._cache.get(
            GraphKind.CONFIG_MODULE_BIPARTITE,
            lambda: views.load_config_module_bipartite(
                self.dataset_root_dir,
                self.repo,
                self.commit,
                use_gpu=self.effective_use_gpu,
            ),
            metadata=metadata,
        )

    def load_config_module_bipartite(self) -> nx.Graph:
        """
        Alias for config_module_bipartite to satisfy GraphEngine protocol.

        Returns
        -------
        nx.Graph
            Config-module bipartite graph.
        """
        return self.config_module_bipartite()

    def _graph_cache_metadata(self, kind: GraphKind) -> GraphCacheMetadata:
        table_keys = _GRAPH_CACHE_TABLES.get(kind, ())
        metadata_by_table = self._parquet_metadata_entries(table_keys)
        build_id = self._collapse_metadata_value(metadata_by_table, "codeintel.build_id")
        schema_hash = self._schema_hash_from_metadata(metadata_by_table)
        repo_meta = self._resolve_metadata_value(metadata_by_table, "codeintel.repo", self.repo)
        commit_meta = self._resolve_metadata_value(
            metadata_by_table,
            "codeintel.commit",
            self.commit,
        )
        return GraphCacheMetadata(
            repo=repo_meta,
            commit=commit_meta,
            build_id=build_id,
            schema_hash=schema_hash,
        )

    def _parquet_metadata_entries(
        self,
        table_keys: tuple[str, ...],
    ) -> dict[str, dict[str, object]]:
        dataset_root = self.dataset_root_dir
        if dataset_root is None or not table_keys:
            return {}
        snapshot_id = self.commit
        entries: dict[str, dict[str, object]] = {}
        for table_key in table_keys:
            metadata = self._parquet_metadata_for_table(dataset_root, table_key, snapshot_id)
            if metadata:
                entries[table_key] = metadata
        return entries

    @staticmethod
    def _parquet_metadata_for_table(
        dataset_root: Path,
        table_key: str,
        snapshot_id: str,
    ) -> dict[str, object] | None:
        try:
            snapshot_dir = dataset_snapshot_dir(
                dataset_root,
                table_key=table_key,
                snapshot_id=snapshot_id,
            )
        except SnapshotIdError as exc:
            log.warning("Invalid snapshot id for Parquet metadata: %s", exc)
            return None
        if not snapshot_dir.exists():
            log.debug("Parquet snapshot missing for %s at %s", table_key, snapshot_dir)
            return None
        try:
            dataset = ds.dataset(str(snapshot_dir), format="parquet", partitioning="hive")
        except (OSError, ValueError, pa.ArrowInvalid) as exc:
            log.debug("Failed to read Parquet metadata for %s: %s", table_key, exc)
            return None
        metadata = metadata_from_schema(dataset.schema)
        if not metadata:
            log.debug("Parquet metadata empty for %s", table_key)
        return metadata

    def _collapse_metadata_value(
        self,
        metadata_by_table: dict[str, dict[str, object]],
        key: str,
    ) -> str | None:
        values = self._metadata_values(metadata_by_table, key)
        if not values:
            return None
        if len(values) == 1:
            return next(iter(values))
        log.warning("Parquet metadata %s differs across tables: %s", key, sorted(values))
        return stable_hash(sorted(values))

    def _resolve_metadata_value(
        self,
        metadata_by_table: dict[str, dict[str, object]],
        key: str,
        fallback: str,
    ) -> str:
        values = self._metadata_values(metadata_by_table, key)
        if not values:
            return fallback
        if len(values) == 1:
            value = next(iter(values))
            if value != fallback:
                log.warning(
                    "Parquet metadata %s mismatch for %s@%s: %s",
                    key,
                    self.repo,
                    self.commit,
                    value,
                )
            return fallback
        log.warning(
            "Parquet metadata %s differs across tables for %s@%s: %s",
            key,
            self.repo,
            self.commit,
            sorted(values),
        )
        return fallback

    @staticmethod
    def _metadata_values(
        metadata_by_table: dict[str, dict[str, object]],
        key: str,
    ) -> set[str]:
        values: set[str] = set()
        for metadata in metadata_by_table.values():
            raw = metadata.get(key)
            if raw is None:
                continue
            if isinstance(raw, str):
                if raw:
                    values.add(raw)
            else:
                values.add(str(raw))
        return values

    @staticmethod
    def _schema_hash_from_metadata(
        metadata_by_table: dict[str, dict[str, object]],
    ) -> str | None:
        schema_by_table: dict[str, str] = {}
        for table_key, metadata in metadata_by_table.items():
            raw = metadata.get("codeintel.schema_hash")
            if isinstance(raw, str) and raw:
                schema_by_table[table_key] = raw
            elif raw is not None:
                schema_by_table[table_key] = str(raw)
        if not schema_by_table:
            return None
        if len(schema_by_table) == 1:
            return next(iter(schema_by_table.values()))
        payload = [
            {"table_key": key, "schema_hash": schema_by_table[key]}
            for key in sorted(schema_by_table)
        ]
        return stable_hash(payload)

    def clear_cache(self) -> None:
        """Clear all cached graphs.

        Forces graphs to be reloaded on next access.
        """
        self._cache.clear()


__all__ = ["NxGraphEngine"]
