"""Unified configuration builder composed from step-specific modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    ScanProfiles,
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
from codeintel.config.steps_ingestion import (
    ConfigIngestStepConfig,
    CoverageIngestStepConfig,
    DocstringStepConfig,
    IngestionStepBuilder,
    PyAstIngestStepConfig,
    RepoScanStepConfig,
    ScipIngestStepConfig,
    TestsIngestStepConfig,
    TypingIngestStepConfig,
)

if TYPE_CHECKING:
    from coverage import Coverage

    from codeintel.config.parser_types import FunctionParserKind
    from codeintel.ingestion.scip_ingest import ScipIngestResult
    from codeintel.ingestion.source_scanner import ScanProfile
    from codeintel.storage.rows import (
        CallGraphEdgeRow,
        CFGBlockRow,
        CFGEdgeRow,
        DFGEdgeRow,
    )


@dataclass
class ConfigBuilder:
    """Build specific step configs from a shared pipeline context."""

    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries = field(default_factory=ToolBinaries)
    profiles: ScanProfiles | None = None
    graph_backend: GraphBackendConfig = field(default_factory=GraphBackendConfig)

    @classmethod
    def from_snapshot(
        cls,
        repo: str,
        commit: str,
        repo_root: Path,
        *,
        build_dir: Path | None = None,
        db_path: Path | None = None,
        document_output_dir: Path | None = None,
        log_db_path: Path | None = None,
        branch: str | None = None,
    ) -> Self:
        """
        Create a builder from basic snapshot parameters.

        Returns
        -------
        Self
            ConfigBuilder ready to produce step configs.
        """
        snapshot = SnapshotRef.from_args(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            branch=branch,
        )
        paths = BuildPaths.from_layout(
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path,
            document_output_dir=document_output_dir,
            log_db_path=log_db_path,
        )
        return cls(snapshot=snapshot, paths=paths)

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
            profiles=profiles,
            graph_backend=graph_backend or GraphBackendConfig(),
        )

    @property
    def ingestion(self) -> IngestionStepBuilder:
        """Access ingestion-related config builders."""
        return IngestionStepBuilder(self)

    @property
    def graphs(self) -> GraphStepBuilder:
        """Access graph-related config builders."""
        return GraphStepBuilder(self)

    @property
    def analytics(self) -> AnalyticsStepBuilder:
        """Access analytics-related config builders."""
        return AnalyticsStepBuilder(self)

    # SCIP and Source Extraction Steps -------------------------------------

    def scip_ingest(
        self,
        *,
        scip_runner: Callable[..., ScipIngestResult] | None = None,
        artifact_writer: Callable[[Path, Path, Path], None] | None = None,
    ) -> ScipIngestStepConfig:
        """
        Build SCIP ingestion configuration.

        Returns
        -------
        ScipIngestStepConfig
            Configuration for SCIP index generation.
        """
        return self.ingestion.scip_ingest(
            scip_runner=scip_runner,
            artifact_writer=artifact_writer,
        )

    def docstring(self) -> DocstringStepConfig:
        """
        Build docstring ingestion configuration.

        Returns
        -------
        DocstringStepConfig
            Configuration for docstring extraction.
        """
        return self.ingestion.docstring()

    def repo_scan(
        self,
        *,
        tags_index_path: Path | None = None,
        tool_runner: object | None = None,
    ) -> RepoScanStepConfig:
        """
        Build repository scan configuration.

        Returns
        -------
        RepoScanStepConfig
            Configuration for module discovery.
        """
        return self.ingestion.repo_scan(
            tags_index_path=tags_index_path,
            tool_runner=tool_runner,
        )

    def coverage_ingest(
        self,
        *,
        coverage_file: Path | None = None,
        tool_runner: object | None = None,
    ) -> CoverageIngestStepConfig:
        """
        Build coverage ingestion configuration.

        Returns
        -------
        CoverageIngestStepConfig
            Configuration for coverage line ingestion.
        """
        return self.ingestion.coverage_ingest(
            coverage_file=coverage_file,
            tool_runner=tool_runner,
        )

    def tests_ingest(
        self,
        *,
        pytest_report_path: Path | None = None,
    ) -> TestsIngestStepConfig:
        """
        Build tests ingestion configuration.

        Returns
        -------
        TestsIngestStepConfig
            Configuration for pytest catalog ingestion.
        """
        return self.ingestion.tests_ingest(pytest_report_path=pytest_report_path)

    def typing_ingest(
        self,
        *,
        tool_runner: object | None = None,
    ) -> TypingIngestStepConfig:
        """
        Build typing ingestion configuration.

        Returns
        -------
        TypingIngestStepConfig
            Configuration for typedness and diagnostics ingestion.
        """
        return self.ingestion.typing_ingest(tool_runner=tool_runner)

    def config_ingest(self) -> ConfigIngestStepConfig:
        """
        Build config-values ingestion configuration.

        Returns
        -------
        ConfigIngestStepConfig
            Configuration for config-values ingestion.
        """
        return self.ingestion.config_ingest()

    def py_ast_ingest(self) -> PyAstIngestStepConfig:
        """
        Build stdlib AST ingestion configuration.

        Returns
        -------
        PyAstIngestStepConfig
            Configuration for AST ingestion.
        """
        return self.ingestion.py_ast_ingest()

    # Graph Construction Steps ---------------------------------------------

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
        return self.graphs.call_graph(
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
        return self.graphs.cfg_builder(cfg_builder=cfg_builder)

    def goid_builder(self, *, language: str = "python") -> GoidBuilderStepConfig:
        """
        Build GOID construction configuration.

        Returns
        -------
        GoidBuilderStepConfig
            Configuration for GOID generation.
        """
        return self.graphs.goid_builder(language=language)

    def import_graph(self) -> ImportGraphStepConfig:
        """
        Build import graph construction configuration.

        Returns
        -------
        ImportGraphStepConfig
            Configuration for import graph construction.
        """
        return self.graphs.import_graph()

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
        return self.graphs.symbol_uses(scip_json_path=scip_json_path)

    # History and Hotspot Steps --------------------------------------------

    def hotspots(self, *, max_commits: int = 2000) -> HotspotsStepConfig:
        """
        Build hotspot scoring configuration.

        Returns
        -------
        HotspotsStepConfig
            Configuration for hotspot analysis.
        """
        return self.analytics.hotspots(max_commits=max_commits)

    def function_history(
        self,
        *,
        max_history_days: int | None = 365,
        min_lines_threshold: int = 1,
        default_branch: str = "HEAD",
    ) -> FunctionHistoryStepConfig:
        """
        Build function history aggregation configuration.

        Returns
        -------
        FunctionHistoryStepConfig
            Configuration for function history aggregation.
        """
        return self.analytics.function_history(
            max_history_days=max_history_days,
            min_lines_threshold=min_lines_threshold,
            default_branch=default_branch,
        )

    def history_timeseries(
        self,
        commits: Sequence[str],
        *,
        entity_kind: str = "function",
        max_entities: int = 500,
        selection_strategy: str = "risk_score",
    ) -> HistoryTimeseriesStepConfig:
        """
        Build cross-commit history timeseries configuration.

        Returns
        -------
        HistoryTimeseriesStepConfig
            Configuration for history timeseries analysis.
        """
        return self.analytics.history_timeseries(
            commits=commits,
            entity_kind=entity_kind,
            max_entities=max_entities,
            selection_strategy=selection_strategy,
        )

    # Coverage and Test Steps ----------------------------------------------

    def coverage_analytics(self) -> CoverageAnalyticsStepConfig:
        """
        Build coverage aggregation configuration.

        Returns
        -------
        CoverageAnalyticsStepConfig
            Configuration for coverage analytics.
        """
        return self.analytics.coverage_analytics()

    def test_coverage(
        self,
        *,
        coverage_file: Path | None = None,
        coverage_loader: Callable[[TestCoverageStepConfig], Coverage | None] | None = None,
    ) -> TestCoverageStepConfig:
        """
        Build test coverage edges configuration.

        Returns
        -------
        TestCoverageStepConfig
            Configuration for test coverage edge extraction.
        """
        return self.analytics.test_coverage(
            coverage_file=coverage_file,
            coverage_loader=coverage_loader,
        )

    def test_profile(
        self,
        *,
        slow_test_threshold_ms: float = 2000.0,
        io_spec: dict[str, object] | None = None,
    ) -> TestProfileStepConfig:
        """
        Build test profile configuration.

        Returns
        -------
        TestProfileStepConfig
            Configuration for test profile analysis.
        """
        return self.analytics.test_profile(
            slow_test_threshold_ms=slow_test_threshold_ms,
            io_spec=io_spec,
        )

    def behavioral_coverage(
        self,
        *,
        enable_llm: bool = False,
        llm_model: str | None = None,
    ) -> BehavioralCoverageStepConfig:
        """
        Build behavioral coverage configuration.

        Returns
        -------
        BehavioralCoverageStepConfig
            Configuration for behavioral coverage analysis.
        """
        return self.analytics.behavioral_coverage(
            enable_llm=enable_llm,
            llm_model=llm_model,
        )

    # Function Analytics Steps ---------------------------------------------

    def function_analytics(
        self,
        *,
        fail_on_missing_spans: bool = False,
        max_workers: int | None = None,
        parser: FunctionParserKind | None = None,
    ) -> FunctionAnalyticsStepConfig:
        """
        Build function metrics configuration.

        Returns
        -------
        FunctionAnalyticsStepConfig
            Configuration for function metrics analysis.
        """
        return self.analytics.function_analytics(
            fail_on_missing_spans=fail_on_missing_spans,
            max_workers=max_workers,
            parser=parser,
        )

    def function_effects(
        self,
        *,
        max_call_depth: int = 3,
        require_all_callees_pure: bool = True,
    ) -> FunctionEffectsStepConfig:
        """
        Build function effects detection configuration.

        Returns
        -------
        FunctionEffectsStepConfig
            Configuration for side-effect detection.
        """
        return self.analytics.function_effects(
            max_call_depth=max_call_depth,
            require_all_callees_pure=require_all_callees_pure,
        )

    def function_contracts(
        self,
        *,
        max_conditions_per_func: int = 64,
    ) -> FunctionContractsStepConfig:
        """
        Build function contracts configuration.

        Returns
        -------
        FunctionContractsStepConfig
            Configuration for contract inference.
        """
        return self.analytics.function_contracts(
            max_conditions_per_func=max_conditions_per_func,
        )

    def semantic_roles(
        self,
        *,
        enable_llm_refinement: bool = False,
    ) -> SemanticRolesStepConfig:
        """
        Build semantic roles configuration.

        Returns
        -------
        SemanticRolesStepConfig
            Configuration for semantic role classification.
        """
        return self.analytics.semantic_roles(
            enable_llm_refinement=enable_llm_refinement,
        )

    # Graph Analytics Steps -------------------------------------------------

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
        return self.graphs.graph_metrics(
            max_betweenness_sample=max_betweenness_sample,
            eigen_max_iter=eigen_max_iter,
            pagerank_weight=pagerank_weight,
            betweenness_weight=betweenness_weight,
            seed=seed,
        )

    # Data Model Steps ------------------------------------------------------

    def data_models(self) -> DataModelsStepConfig:
        """
        Build data model extraction configuration.

        Returns
        -------
        DataModelsStepConfig
            Configuration for data model extraction.
        """
        return self.analytics.data_models()

    def data_model_usage(
        self,
        *,
        max_examples_per_usage: int = 5,
    ) -> DataModelUsageStepConfig:
        """
        Build data model usage analytics configuration.

        Returns
        -------
        DataModelUsageStepConfig
            Configuration for data model usage analysis.
        """
        return self.analytics.data_model_usage(
            max_examples_per_usage=max_examples_per_usage,
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
        return self.graphs.config_data_flow(
            max_paths_per_usage=max_paths_per_usage,
            max_path_length=max_path_length,
        )

    # Entrypoint and Dependency Steps --------------------------------------

    def entrypoints(
        self,
        *,
        scan_profile: ScanProfile | None = None,
        toggles: EntryPointToggles | None = None,
    ) -> EntryPointsStepConfig:
        """
        Build entrypoint detection configuration.

        Returns
        -------
        EntryPointsStepConfig
            Configuration for entrypoint detection.
        """
        return self.analytics.entrypoints(
            scan_profile=scan_profile,
            toggles=toggles,
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
        return self.graphs.external_dependencies(
            language=language,
            dependency_patterns_path=dependency_patterns_path,
            scan_profile=scan_profile,
        )

    # Aggregation and Profile Steps ----------------------------------------

    def profiles_analytics(self) -> ProfilesAnalyticsStepConfig:
        """
        Build profiles analytics configuration.

        Returns
        -------
        ProfilesAnalyticsStepConfig
            Configuration for profiles analysis.
        """
        return self.analytics.profiles_analytics()

    def subsystems(
        self,
        *,
        min_modules: int = 3,
        max_subsystems: int | None = None,
        import_weight: float = 1.0,
        symbol_weight: float = 0.5,
        config_weight: float = 0.3,
    ) -> SubsystemsStepConfig:
        """
        Build subsystems inference configuration.

        Returns
        -------
        SubsystemsStepConfig
            Configuration for subsystem inference.
        """
        return self.analytics.subsystems(
            min_modules=min_modules,
            max_subsystems=max_subsystems,
            import_weight=import_weight,
            symbol_weight=symbol_weight,
            config_weight=config_weight,
        )


__all__ = [
    "BehavioralCoverageStepConfig",
    "CFGBuilderStepConfig",
    "CallGraphStepConfig",
    "ConfigBuilder",
    "ConfigDataFlowStepConfig",
    "ConfigIngestStepConfig",
    "CoverageAnalyticsStepConfig",
    "CoverageIngestStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "DocstringStepConfig",
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
    "PyAstIngestStepConfig",
    "RepoScanStepConfig",
    "ScipIngestStepConfig",
    "SemanticRolesStepConfig",
    "SubsystemsStepConfig",
    "SymbolUsesStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
    "TestsIngestStepConfig",
    "TypingIngestStepConfig",
]
