"""Compatibility layer for migrating between old and new configuration systems.

This module provides converters between the legacy step configs in `models.py`
and the new composition-based step configs in `builder.py`. Use these during
the transition period; direct use of `ConfigBuilder` is preferred for new code.

Migration Guide
---------------
Old code:
    from codeintel.config.models import GraphMetricsConfig
    cfg = GraphMetricsConfig.from_paths(repo="r", commit="c")

New code:
    from codeintel.config import ConfigBuilder
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path("."))
    cfg = builder.graph_metrics()

During transition (this module):
    from codeintel.config.compat import to_graph_metrics_step
    old_cfg = GraphMetricsConfig.from_paths(repo="r", commit="c")
    new_cfg = to_graph_metrics_step(old_cfg, repo_root=Path("."))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.builder import (
    BehavioralCoverageStepConfig,
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    ConfigDataFlowStepConfig,
    CoverageAnalyticsStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    DocstringStepConfig,
    EntryPointsStepConfig,
    ExternalDependenciesStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
    GoidBuilderStepConfig,
    GraphMetricsStepConfig,
    HistoryTimeseriesStepConfig,
    HotspotsStepConfig,
    ImportGraphStepConfig,
    ProfilesAnalyticsStepConfig,
    ScipIngestStepConfig,
    SemanticRolesStepConfig,
    SubsystemsStepConfig,
    SymbolUsesStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.primitives import BuildPaths, SnapshotRef, ToolBinaries

if TYPE_CHECKING:
    from codeintel.config.models import (
        BehavioralCoverageConfig,
        CallGraphConfig,
        CFGBuilderConfig,
        ConfigDataFlowConfig,
        CoverageAnalyticsConfig,
        DataModelsConfig,
        DataModelUsageConfig,
        DocstringConfig,
        EntryPointsConfig,
        ExternalDependenciesConfig,
        FunctionAnalyticsConfig,
        FunctionContractsConfig,
        FunctionEffectsConfig,
        FunctionHistoryConfig,
        GoidBuilderConfig,
        GraphMetricsConfig,
        HistoryTimeseriesConfig,
        HotspotsConfig,
        ImportGraphConfig,
        ProfilesAnalyticsConfig,
        ScipIngestConfig,
        SemanticRolesConfig,
        SubsystemsConfig,
        SymbolUsesConfig,
        TestCoverageConfig,
        TestProfileConfig,
        ToolsConfig,
    )


def _snapshot_from_legacy(
    repo: str,
    commit: str,
    repo_root: Path,
) -> SnapshotRef:
    """Create a SnapshotRef from legacy config fields."""
    return SnapshotRef.from_args(repo=repo, commit=commit, repo_root=repo_root)


def _paths_from_legacy(
    repo_root: Path,
    build_dir: Path | None = None,
) -> BuildPaths:
    """Create BuildPaths from legacy config fields."""
    return BuildPaths.from_repo_root(repo_root, build_dir=build_dir)


def _binaries_from_tools(tools: ToolsConfig | None) -> ToolBinaries:
    """Create ToolBinaries from legacy ToolsConfig."""
    if tools is None:
        return ToolBinaries()
    return ToolBinaries(
        scip_python_bin=tools.scip_python_bin,
        scip_bin=tools.scip_bin,
        pyright_bin=tools.pyright_bin,
        pyrefly_bin=tools.pyrefly_bin,
        ruff_bin=tools.ruff_bin,
        coverage_bin=tools.coverage_bin,
        pytest_bin=tools.pytest_bin,
        git_bin=tools.git_bin,
        default_timeout_s=tools.default_timeout_s,
    )


# ---------------------------------------------------------------------------
# Converters: Legacy Config -> New StepConfig
# ---------------------------------------------------------------------------


def to_scip_ingest_step(
    cfg: ScipIngestConfig,
    tools: ToolsConfig | None = None,
) -> ScipIngestStepConfig:
    """Convert legacy ScipIngestConfig to ScipIngestStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    paths = BuildPaths.from_explicit(
        build_dir=cfg.build_dir,
        document_output_dir=cfg.document_output_dir,
    )
    binaries = ToolBinaries(
        scip_python_bin=cfg.scip_python_bin,
        scip_bin=cfg.scip_bin,
    )
    if tools is not None:
        binaries = _binaries_from_tools(tools)
    return ScipIngestStepConfig(
        snapshot=snapshot,
        paths=paths,
        binaries=binaries,
        scip_runner=cfg.scip_runner,
        artifact_writer=cfg.artifact_writer,
    )


def to_docstring_step(cfg: DocstringConfig) -> DocstringStepConfig:
    """Convert legacy DocstringConfig to DocstringStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return DocstringStepConfig(snapshot=snapshot)


def to_call_graph_step(cfg: CallGraphConfig) -> CallGraphStepConfig:
    """Convert legacy CallGraphConfig to CallGraphStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return CallGraphStepConfig(
        snapshot=snapshot,
        cst_collector=cfg.cst_collector,
        ast_collector=cfg.ast_collector,
    )


def to_cfg_builder_step(cfg: CFGBuilderConfig) -> CFGBuilderStepConfig:
    """Convert legacy CFGBuilderConfig to CFGBuilderStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return CFGBuilderStepConfig(
        snapshot=snapshot,
        cfg_builder=cfg.cfg_builder,
    )


def to_goid_builder_step(cfg: GoidBuilderConfig) -> GoidBuilderStepConfig:
    """Convert legacy GoidBuilderConfig to GoidBuilderStepConfig.

    Note: GoidBuilderConfig doesn't have repo_root, so we use a placeholder.
    """
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=Path())
    return GoidBuilderStepConfig(
        snapshot=snapshot,
        language=cfg.language,
    )


def to_import_graph_step(cfg: ImportGraphConfig) -> ImportGraphStepConfig:
    """Convert legacy ImportGraphConfig to ImportGraphStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return ImportGraphStepConfig(snapshot=snapshot)


def to_symbol_uses_step(
    cfg: SymbolUsesConfig,
    build_dir: Path | None = None,
) -> SymbolUsesStepConfig:
    """Convert legacy SymbolUsesConfig to SymbolUsesStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    paths = _paths_from_legacy(cfg.repo_root, build_dir=build_dir)
    return SymbolUsesStepConfig(
        snapshot=snapshot,
        paths=paths,
        scip_json_path=cfg.scip_json_path,
    )


def to_hotspots_step(cfg: HotspotsConfig) -> HotspotsStepConfig:
    """Convert legacy HotspotsConfig to HotspotsStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return HotspotsStepConfig(
        snapshot=snapshot,
        max_commits=cfg.max_commits,
    )


def to_function_history_step(cfg: FunctionHistoryConfig) -> FunctionHistoryStepConfig:
    """Convert legacy FunctionHistoryConfig to FunctionHistoryStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return FunctionHistoryStepConfig(
        snapshot=snapshot,
        max_history_days=cfg.max_history_days,
        min_lines_threshold=cfg.min_lines_threshold,
        default_branch=cfg.default_branch,
    )


def to_history_timeseries_step(
    cfg: HistoryTimeseriesConfig,
) -> HistoryTimeseriesStepConfig:
    """Convert legacy HistoryTimeseriesConfig to HistoryTimeseriesStepConfig."""
    snapshot = SnapshotRef(
        repo=cfg.repo,
        commit=cfg.commits[0] if cfg.commits else "",
        repo_root=cfg.repo_root,
    )
    return HistoryTimeseriesStepConfig(
        snapshot=snapshot,
        commits=cfg.commits,
        entity_kind=cfg.entity_kind,
        max_entities=cfg.max_entities,
        selection_strategy=cfg.selection_strategy,
    )


def to_coverage_analytics_step(cfg: CoverageAnalyticsConfig) -> CoverageAnalyticsStepConfig:
    """Convert legacy CoverageAnalyticsConfig to CoverageAnalyticsStepConfig."""
    snapshot = SnapshotRef(repo=cfg.repo, commit=cfg.commit, repo_root=Path())
    return CoverageAnalyticsStepConfig(snapshot=snapshot)


def to_function_analytics_step(cfg: FunctionAnalyticsConfig) -> FunctionAnalyticsStepConfig:
    """Convert legacy FunctionAnalyticsConfig to FunctionAnalyticsStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return FunctionAnalyticsStepConfig(
        snapshot=snapshot,
        fail_on_missing_spans=cfg.fail_on_missing_spans,
        max_workers=cfg.max_workers,
        parser=cfg.parser,
    )


def to_graph_metrics_step(
    cfg: GraphMetricsConfig,
    repo_root: Path | None = None,
) -> GraphMetricsStepConfig:
    """Convert legacy GraphMetricsConfig to GraphMetricsStepConfig."""
    snapshot = SnapshotRef(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=repo_root or Path(),
    )
    return GraphMetricsStepConfig(
        snapshot=snapshot,
        max_betweenness_sample=cfg.max_betweenness_sample,
        eigen_max_iter=cfg.eigen_max_iter,
        pagerank_weight=cfg.pagerank_weight,
        betweenness_weight=cfg.betweenness_weight,
        seed=cfg.seed,
    )


def to_data_models_step(cfg: DataModelsConfig) -> DataModelsStepConfig:
    """Convert legacy DataModelsConfig to DataModelsStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return DataModelsStepConfig(snapshot=snapshot)


def to_data_model_usage_step(cfg: DataModelUsageConfig) -> DataModelUsageStepConfig:
    """Convert legacy DataModelUsageConfig to DataModelUsageStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return DataModelUsageStepConfig(
        snapshot=snapshot,
        max_examples_per_usage=cfg.max_examples_per_usage,
    )


def to_config_data_flow_step(cfg: ConfigDataFlowConfig) -> ConfigDataFlowStepConfig:
    """Convert legacy ConfigDataFlowConfig to ConfigDataFlowStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return ConfigDataFlowStepConfig(
        snapshot=snapshot,
        max_paths_per_usage=cfg.max_paths_per_usage,
        max_path_length=cfg.max_path_length,
    )


def to_entrypoints_step(cfg: EntryPointsConfig) -> EntryPointsStepConfig:
    """Convert legacy EntryPointsConfig to EntryPointsStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return EntryPointsStepConfig(
        snapshot=snapshot,
        scan_profile=cfg.scan_profile,
        detect_fastapi=cfg.detect_fastapi,
        detect_flask=cfg.detect_flask,
        detect_click=cfg.detect_click,
        detect_typer=cfg.detect_typer,
        detect_cron=cfg.detect_cron,
        detect_django=cfg.detect_django,
        detect_celery=cfg.detect_celery,
        detect_airflow=cfg.detect_airflow,
        detect_generic_routes=cfg.detect_generic_routes,
    )


def to_external_dependencies_step(
    cfg: ExternalDependenciesConfig,
) -> ExternalDependenciesStepConfig:
    """Convert legacy ExternalDependenciesConfig to ExternalDependenciesStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return ExternalDependenciesStepConfig(
        snapshot=snapshot,
        language=cfg.language,
        dependency_patterns_path=cfg.dependency_patterns_path,
        scan_profile=cfg.scan_profile,
    )


def to_function_effects_step(cfg: FunctionEffectsConfig) -> FunctionEffectsStepConfig:
    """Convert legacy FunctionEffectsConfig to FunctionEffectsStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return FunctionEffectsStepConfig(
        snapshot=snapshot,
        max_call_depth=cfg.max_call_depth,
        require_all_callees_pure=cfg.require_all_callees_pure,
        io_apis=dict(cfg.io_apis),
        db_apis=dict(cfg.db_apis),
        time_apis=dict(cfg.time_apis),
        random_apis=dict(cfg.random_apis),
        threading_apis=dict(cfg.threading_apis),
    )


def to_function_contracts_step(cfg: FunctionContractsConfig) -> FunctionContractsStepConfig:
    """Convert legacy FunctionContractsConfig to FunctionContractsStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return FunctionContractsStepConfig(
        snapshot=snapshot,
        max_conditions_per_func=cfg.max_conditions_per_func,
    )


def to_semantic_roles_step(cfg: SemanticRolesConfig) -> SemanticRolesStepConfig:
    """Convert legacy SemanticRolesConfig to SemanticRolesStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return SemanticRolesStepConfig(
        snapshot=snapshot,
        enable_llm_refinement=cfg.enable_llm_refinement,
    )


def to_profiles_analytics_step(
    cfg: ProfilesAnalyticsConfig,
    repo_root: Path | None = None,
) -> ProfilesAnalyticsStepConfig:
    """Convert legacy ProfilesAnalyticsConfig to ProfilesAnalyticsStepConfig."""
    snapshot = SnapshotRef(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=repo_root or Path(),
    )
    return ProfilesAnalyticsStepConfig(snapshot=snapshot)


def to_subsystems_step(
    cfg: SubsystemsConfig,
    repo_root: Path | None = None,
) -> SubsystemsStepConfig:
    """Convert legacy SubsystemsConfig to SubsystemsStepConfig."""
    snapshot = SnapshotRef(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=repo_root or Path(),
    )
    return SubsystemsStepConfig(
        snapshot=snapshot,
        min_modules=cfg.min_modules,
        max_subsystems=cfg.max_subsystems,
        import_weight=cfg.import_weight,
        symbol_weight=cfg.symbol_weight,
        config_weight=cfg.config_weight,
    )


def to_test_coverage_step(cfg: TestCoverageConfig) -> TestCoverageStepConfig:
    """Convert legacy TestCoverageConfig to TestCoverageStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return TestCoverageStepConfig(
        snapshot=snapshot,
        coverage_file=cfg.coverage_file,
    )


def to_test_profile_step(cfg: TestProfileConfig) -> TestProfileStepConfig:
    """Convert legacy TestProfileConfig to TestProfileStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return TestProfileStepConfig(
        snapshot=snapshot,
        slow_test_threshold_ms=cfg.slow_test_threshold_ms,
        io_spec=cfg.io_spec,
    )


def to_behavioral_coverage_step(
    cfg: BehavioralCoverageConfig,
) -> BehavioralCoverageStepConfig:
    """Convert legacy BehavioralCoverageConfig to BehavioralCoverageStepConfig."""
    snapshot = _snapshot_from_legacy(cfg.repo, cfg.commit, cfg.repo_root)
    return BehavioralCoverageStepConfig(
        snapshot=snapshot,
        heuristic_version=cfg.heuristic_version,
        enable_llm=cfg.enable_llm,
        llm_model=cfg.llm_model,
    )


__all__ = [
    # Converters
    "to_behavioral_coverage_step",
    "to_call_graph_step",
    "to_cfg_builder_step",
    "to_config_data_flow_step",
    "to_coverage_analytics_step",
    "to_data_model_usage_step",
    "to_data_models_step",
    "to_docstring_step",
    "to_entrypoints_step",
    "to_external_dependencies_step",
    "to_function_analytics_step",
    "to_function_contracts_step",
    "to_function_effects_step",
    "to_function_history_step",
    "to_goid_builder_step",
    "to_graph_metrics_step",
    "to_history_timeseries_step",
    "to_hotspots_step",
    "to_import_graph_step",
    "to_profiles_analytics_step",
    "to_scip_ingest_step",
    "to_semantic_roles_step",
    "to_subsystems_step",
    "to_symbol_uses_step",
    "to_test_coverage_step",
    "to_test_profile_step",
]
