"""Unified configuration builder composed from step-specific modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self, overload

from codeintel.config.primitives import (
    BuildLayoutOptions,
    BuildPaths,
    GraphBackendConfig,
    ScanProfiles,
    SnapshotInit,
    SnapshotRef,
    ToolBinaries,
)
from codeintel.config.steps_analytics import (
    AnalyticsStepBuilder,
    BehavioralCoverageStepConfig,
    CoverageAnalyticsStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    EntryPointsStepConfig,
    EntryPointToggles,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
    HistoryTimeseriesStepConfig,
    HotspotsStepConfig,
    ProfilesAnalyticsStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.steps_graphs import (
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    ConfigDataFlowStepConfig,
    ExternalDependenciesStepConfig,
    GoidBuilderStepConfig,
    GraphMetricsStepConfig,
    GraphStepBuilder,
    ImportGraphStepConfig,
    SymbolUsesStepConfig,
)


@dataclass(frozen=True)
class BuilderDependencies:
    """Optional overrides for builder-scoped dependencies."""

    binaries: ToolBinaries | None = None
    profiles: ScanProfiles | None = None
    graph_backend: GraphBackendConfig | None = None

    def resolved(self) -> tuple[ToolBinaries, ScanProfiles | None, GraphBackendConfig]:
        """Return dependency instances with defaults applied.

        Returns
        -------
        tuple[ToolBinaries, ScanProfiles | None, GraphBackendConfig]
            Concrete binaries, scan profiles, and graph backend configuration.
        """
        return (
            self.binaries or ToolBinaries(),
            self.profiles,
            self.graph_backend or GraphBackendConfig(),
        )


@dataclass
class ConfigBuilder:
    """Build specific step configs from a shared pipeline context.

    Explicit facets (`graphs`, `analytics`) are preferred; legacy step helpers are
    still available via attribute delegation for compatibility.
    """

    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries = field(default_factory=ToolBinaries)
    profiles: ScanProfiles | None = None
    graph_backend: GraphBackendConfig = field(default_factory=GraphBackendConfig)
    _GRAPH_DELEGATES: ClassVar[frozenset[str]] = frozenset(
        {
            "call_graph",
            "cfg_builder",
            "goid_builder",
            "import_graph",
            "symbol_uses",
            "graph_metrics",
            "config_data_flow",
            "external_dependencies",
        }
    )
    _ANALYTICS_DELEGATES: ClassVar[frozenset[str]] = frozenset(
        {
            "hotspots",
            "function_history",
            "history_timeseries",
            "coverage_analytics",
            "test_coverage",
            "test_profile",
            "behavioral_coverage",
            "function_analytics",
            "function_effects",
            "function_contracts",
            "semantic_roles",
            "data_models",
            "data_model_usage",
            "profiles_analytics",
            "subsystems",
            "entrypoints",
        }
    )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SnapshotInit,
        *,
        layout: BuildLayoutOptions | None = None,
        primitives: BuilderDependencies | None = None,
    ) -> Self:
        """Create a builder from snapshot and layout primitives.

        Parameters
        ----------
        snapshot
            Snapshot parameters (repo, commit, repo_root, optional branch).
        layout
            Optional build layout overrides.
        primitives
            Optional dependency overrides (binaries, profiles, graph backend).

        Returns
        -------
        Self
            ConfigBuilder ready to produce step configs.
        """
        layout_options = layout or BuildLayoutOptions()
        dependencies = primitives or BuilderDependencies()
        snapshot_ref = snapshot.to_snapshot_ref()
        has_layout_overrides = any(
            value is not None
            for value in (
                layout_options.build_dir,
                layout_options.db_path,
                layout_options.document_output_dir,
                layout_options.log_db_path,
            )
        )
        paths = layout_options.materialize(
            snapshot_ref.repo_root,
            check_collisions=has_layout_overrides,
        )
        binaries, profiles, graph_backend = dependencies.resolved()
        profiles = cls._ensure_profiles(profiles)
        return cls(
            snapshot=snapshot_ref,
            paths=paths,
            binaries=binaries,
            profiles=profiles,
            graph_backend=graph_backend,
        )

    @classmethod
    def from_primitives(
        cls,
        snapshot: SnapshotRef,
        paths: BuildPaths,
        *,
        binaries: ToolBinaries | None = None,
        profiles: ScanProfiles | None = None,
        graph_backend: GraphBackendConfig | None = None,
    ) -> Self:
        """
        Create a builder from pre-constructed primitives.

        Returns
        -------
        Self
            ConfigBuilder ready to produce step configs.
        """
        return cls(
            snapshot=snapshot,
            paths=paths,
            binaries=binaries or ToolBinaries(),
            profiles=cls._ensure_profiles(profiles),
            graph_backend=graph_backend or GraphBackendConfig(),
        )

    @property
    def graphs(self) -> GraphStepBuilder:
        """Access graph-related config builders."""
        return GraphStepBuilder(self)

    @property
    def analytics(self) -> AnalyticsStepBuilder:
        """Access analytics-related config builders."""
        return AnalyticsStepBuilder(self)

    if TYPE_CHECKING:

        @overload
        def __getattr__(
            self, name: Literal["call_graph"]
        ) -> Callable[..., CallGraphStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["cfg_builder"]
        ) -> Callable[..., CFGBuilderStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["goid_builder"]
        ) -> Callable[..., GoidBuilderStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["import_graph"]
        ) -> Callable[..., ImportGraphStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["symbol_uses"]
        ) -> Callable[..., SymbolUsesStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["graph_metrics"]
        ) -> Callable[..., GraphMetricsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["config_data_flow"]
        ) -> Callable[..., ConfigDataFlowStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["external_dependencies"]
        ) -> Callable[..., ExternalDependenciesStepConfig]: ...

        @overload
        def __getattr__(self, name: Literal["hotspots"]) -> Callable[..., HotspotsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["function_history"]
        ) -> Callable[..., FunctionHistoryStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["history_timeseries"]
        ) -> Callable[..., HistoryTimeseriesStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["coverage_analytics"]
        ) -> Callable[..., CoverageAnalyticsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["test_coverage"]
        ) -> Callable[..., TestCoverageStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["test_profile"]
        ) -> Callable[..., TestProfileStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["behavioral_coverage"]
        ) -> Callable[..., BehavioralCoverageStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["function_analytics"]
        ) -> Callable[..., FunctionAnalyticsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["function_effects"]
        ) -> Callable[..., FunctionEffectsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["function_contracts"]
        ) -> Callable[..., FunctionContractsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["semantic_roles"]
        ) -> Callable[..., SemanticRolesStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["data_models"]
        ) -> Callable[..., DataModelsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["data_model_usage"]
        ) -> Callable[..., DataModelUsageStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["profiles_analytics"]
        ) -> Callable[..., ProfilesAnalyticsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["subsystems"]
        ) -> Callable[..., SubsystemsStepConfig]: ...

        @overload
        def __getattr__(
            self, name: Literal["entrypoints"]
        ) -> Callable[..., EntryPointsStepConfig]: ...

        @overload
        def __getattr__(self, name: str) -> object: ...

    def __getattr__(self, name: str) -> object:
        """Delegate legacy step helpers to the underlying facet builders.

        Parameters
        ----------
        name
            Requested attribute name.

        Returns
        -------
        object
            Delegated step builder callable.

        Raises
        ------
        AttributeError
            If the attribute cannot be resolved from delegated builders.
        """
        if name in self._GRAPH_DELEGATES:
            return getattr(self.graphs, name)
        if name in self._ANALYTICS_DELEGATES:
            return getattr(self.analytics, name)
        message = f"{type(self).__name__!s} has no attribute {name!r}"
        raise AttributeError(message)

    @staticmethod
    def _ensure_profiles(profiles: ScanProfiles | None) -> ScanProfiles | None:
        """Validate scan profiles and enforce completeness when provided.

        Parameters
        ----------
        profiles
            Optional scan profiles bundle.

        Returns
        -------
        ScanProfiles | None
            Validated profiles or None.

        Raises
        ------
        TypeError
            If provided profiles are not a ScanProfiles instance.
        ValueError
            If provided profiles are missing code or config entries.
        """
        if profiles is None:
            return None
        if not isinstance(profiles, ScanProfiles):
            message = "profiles must be a ScanProfiles instance when provided"
            raise TypeError(message)
        if profiles.code is None or profiles.config is None:
            message = "profiles must include both code and config scan profiles"
            raise ValueError(message)
        return profiles

    def __dir__(self) -> list[str]:
        """Expose delegated step names for discoverability.

        Returns
        -------
        list[str]
            Combined attributes from the base class and delegated helpers.
        """
        base = super().__dir__()
        dynamic = (*self._GRAPH_DELEGATES, *self._ANALYTICS_DELEGATES)
        return sorted(set(base).union(dynamic))

    def prepare_filesystem(self, *, create_missing_only: bool = True) -> tuple[Path, ...]:
        """Ensure build-related directories exist.

        Parameters
        ----------
        create_missing_only
            When True, create only directories that do not already exist.

        Returns
        -------
        tuple[Path, ...]
            Directories created during preparation.
        """
        targets = (
            self.paths.build_dir,
            self.paths.db_path.parent,
            self.paths.document_output_dir,
            self.paths.scip_dir,
            self.paths.coverage_json.parent,
            self.paths.pytest_report.parent,
            self.paths.tool_cache,
            self.paths.log_db_path.parent,
        )
        created: list[Path] = []
        for target in targets:
            if create_missing_only and target.exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)
        return tuple(created)


__all__ = [
    "BehavioralCoverageStepConfig",
    "CFGBuilderStepConfig",
    "CallGraphStepConfig",
    "ConfigBuilder",
    "ConfigDataFlowStepConfig",
    "CoverageAnalyticsStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "EntryPointToggles",
    "EntryPointsStepConfig",
    "ExternalDependenciesStepConfig",
    "FunctionAnalyticsStepConfig",
    "FunctionContractsStepConfig",
    "FunctionEffectsStepConfig",
    "FunctionHistoryStepConfig",
    "GoidBuilderStepConfig",
    "GraphMetricsStepConfig",
    "HistoryTimeseriesStepConfig",
    "HotspotsStepConfig",
    "ImportGraphStepConfig",
    "ProfilesAnalyticsStepConfig",
    "SemanticRolesStepConfig",
    "SubsystemsStepConfig",
    "SymbolUsesStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
]
