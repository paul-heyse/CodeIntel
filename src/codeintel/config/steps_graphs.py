"""Graph-related step configuration models and builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from codeintel.config.primitives import BuildPaths, ScanProfiles, SnapshotRef

if TYPE_CHECKING:
    from codeintel.storage.rows import (
        CallGraphEdgeRow,
        CFGBlockRow,
        CFGEdgeRow,
        DFGEdgeRow,
    )
    from codeintel.ingestion.source_scanner import ScanProfile


@dataclass(frozen=True)
class CallGraphStepConfig:
    """Configuration for call graph construction."""

    snapshot: SnapshotRef
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class CFGBuilderStepConfig:
    """Configuration for control and data flow graph generation."""

    snapshot: SnapshotRef
    cfg_builder: (
        Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
    ) = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class GoidBuilderStepConfig:
    """Configuration for GOID generation."""

    snapshot: SnapshotRef
    language: str = "python"

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class ImportGraphStepConfig:
    """Configuration for import graph construction."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class SymbolUsesStepConfig:
    """Configuration for symbol use derivation."""

    snapshot: SnapshotRef
    paths: BuildPaths
    scip_json_path: Path | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def resolved_scip_json_path(self) -> Path:
        """Path to the SCIP JSON file."""
        return self.scip_json_path or self.paths.build_dir / "scip" / "index.scip.json"


@dataclass(frozen=True)
class GraphMetricsStepConfig:
    """Configuration for graph metrics analytics."""

    snapshot: SnapshotRef
    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    seed: int = 0

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class ConfigDataFlowStepConfig:
    """Configuration for config data flow analytics."""

    snapshot: SnapshotRef
    max_paths_per_usage: int = 3
    max_path_length: int = 10

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class ExternalDependenciesStepConfig:
    """Configuration for external dependency analytics."""

    snapshot: SnapshotRef
    language: str = "python"
    dependency_patterns_path: Path | None = None
    scan_profile: ScanProfile | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


class _GraphOwner(Protocol):
    snapshot: SnapshotRef
    paths: BuildPaths
    profiles: ScanProfiles | None


class GraphStepBuilder:
    """Graph and dependency related config builders."""

    def __init__(self, owner: _GraphOwner) -> None:
        self._owner = owner

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference shared across graph configs."""
        return self._owner.snapshot

    @property
    def paths(self) -> BuildPaths:
        """Build paths used for graph construction outputs."""
        return self._owner.paths

    @property
    def profiles(self) -> ScanProfiles | None:
        """Optional scan profiles supplied by the owning builder."""
        return getattr(self._owner, "profiles", None)

    def call_graph(
        self,
        *,
        cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None,
        ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None,
    ) -> CallGraphStepConfig:
        """
        Build call graph construction configuration.

        Returns
        -------
        CallGraphStepConfig
            Configuration for call graph construction.
        """
        return CallGraphStepConfig(
            snapshot=self.snapshot,
            cst_collector=cst_collector,
            ast_collector=ast_collector,
        )

    def cfg_builder(
        self,
        *,
        cfg_builder: (
            Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
        ) = None,
    ) -> CFGBuilderStepConfig:
        """
        Build CFG/DFG scaffolding configuration.

        Returns
        -------
        CFGBuilderStepConfig
            Configuration for control/data flow graph construction.
        """
        return CFGBuilderStepConfig(
            snapshot=self.snapshot,
            cfg_builder=cfg_builder,
        )

    def goid_builder(self, *, language: str = "python") -> GoidBuilderStepConfig:
        """
        Build GOID construction configuration.

        Returns
        -------
        GoidBuilderStepConfig
            Configuration for GOID generation.
        """
        return GoidBuilderStepConfig(
            snapshot=self.snapshot,
            language=language,
        )

    def import_graph(self) -> ImportGraphStepConfig:
        """
        Build import graph construction configuration.

        Returns
        -------
        ImportGraphStepConfig
            Configuration for import graph construction.
        """
        return ImportGraphStepConfig(snapshot=self.snapshot)

    def symbol_uses(
        self,
        *,
        scip_json_path: Path | None = None,
    ) -> SymbolUsesStepConfig:
        """
        Build symbol uses derivation configuration.

        Returns
        -------
        SymbolUsesStepConfig
            Configuration for symbol usage extraction.
        """
        return SymbolUsesStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            scip_json_path=scip_json_path,
        )

    def graph_metrics(
        self,
        *,
        max_betweenness_sample: int | None = 200,
        eigen_max_iter: int = 200,
        pagerank_weight: str | None = "weight",
        betweenness_weight: str | None = "weight",
        seed: int = 0,
    ) -> GraphMetricsStepConfig:
        """
        Build graph metrics analytics configuration.

        Returns
        -------
        GraphMetricsStepConfig
            Configuration for graph centrality metrics.
        """
        return GraphMetricsStepConfig(
            snapshot=self.snapshot,
            max_betweenness_sample=max_betweenness_sample,
            eigen_max_iter=eigen_max_iter,
            pagerank_weight=pagerank_weight,
            betweenness_weight=betweenness_weight,
            seed=seed,
        )

    def config_data_flow(
        self,
        *,
        max_paths_per_usage: int = 3,
        max_path_length: int = 10,
    ) -> ConfigDataFlowStepConfig:
        """
        Build config data flow analytics configuration.

        Returns
        -------
        ConfigDataFlowStepConfig
            Configuration for config data flow analysis.
        """
        return ConfigDataFlowStepConfig(
            snapshot=self.snapshot,
            max_paths_per_usage=max_paths_per_usage,
            max_path_length=max_path_length,
        )

    def external_dependencies(
        self,
        *,
        language: str = "python",
        dependency_patterns_path: Path | None = None,
        scan_profile: ScanProfile | None = None,
    ) -> ExternalDependenciesStepConfig:
        """
        Build external dependencies configuration.

        Returns
        -------
        ExternalDependenciesStepConfig
            Configuration for dependency analysis.
        """
        profile = scan_profile
        if profile is None and self.profiles is not None:
            profile = self.profiles.code
        return ExternalDependenciesStepConfig(
            snapshot=self.snapshot,
            language=language,
            dependency_patterns_path=dependency_patterns_path,
            scan_profile=profile,
        )


__all__ = [
    "CFGBuilderStepConfig",
    "CallGraphStepConfig",
    "ConfigDataFlowStepConfig",
    "ExternalDependenciesStepConfig",
    "GoidBuilderStepConfig",
    "GraphMetricsStepConfig",
    "GraphStepBuilder",
    "ImportGraphStepConfig",
    "SymbolUsesStepConfig",
]
