"""Integration-style tests for analytics compute modules."""

from __future__ import annotations

from codeintel.analytics.data_models.core import compute_data_models
from codeintel.analytics.dependencies.core import (
    ExternalDependencyInputs,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints.core import build_entrypoints
from codeintel.analytics.functions.function_contracts import compute_function_contracts
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    compute_function_effects,
)
from codeintel.analytics.functions.function_history import compute_function_history
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.config.steps_analytics import (
    DataModelsStepConfig,
    EntryPointsStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
)
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig
from codeintel.graphs.catalog import FunctionCatalogService
from tests.analytics.integration.sample_repo import (
    SampleRepo,
    build_runtime,
    count_table_rows,
)


def test_full_analytics_pipeline(sample_repo: SampleRepo) -> None:
    """Execute compute flows end-to-end on a small in-memory snapshot."""
    catalog = FunctionCatalogService.from_db(
        sample_repo.gateway,
        repo=sample_repo.snapshot.repo,
        commit=sample_repo.snapshot.commit,
    )

    summary = compute_function_metrics_and_types(
        sample_repo.gateway,
        FunctionAnalyticsStepConfig(snapshot=sample_repo.snapshot),
    )
    assert summary["metrics_rows"] >= 2

    compute_function_contracts(
        sample_repo.gateway,
        FunctionContractsStepConfig(snapshot=sample_repo.snapshot, max_conditions_per_func=5),
        function_ast_map=sample_repo.ast_map,
        catalog=catalog,
    )
    assert count_table_rows(sample_repo, "analytics.function_contracts") >= 2

    runtime = build_runtime(sample_repo)
    compute_function_effects(
        sample_repo.gateway,
        FunctionEffectsStepConfig(snapshot=sample_repo.snapshot),
        inputs=FunctionEffectsInputs(
            catalog_provider=catalog,
            runtime=runtime,
            ast_map=sample_repo.ast_map,
            missing_goids=set(),
        ),
    )
    assert count_table_rows(sample_repo, "analytics.function_effects") >= 2

    compute_function_history(
        sample_repo.gateway,
        FunctionHistoryStepConfig(
            snapshot=sample_repo.snapshot,
            max_history_days=7,
            min_lines_threshold=1,
            default_branch="HEAD",
        ),
    )
    assert count_table_rows(sample_repo, "analytics.function_history") >= 2

    compute_data_models(
        sample_repo.gateway,
        DataModelsStepConfig(snapshot=sample_repo.snapshot),
    )
    assert count_table_rows(sample_repo, "analytics.data_models") >= 1

    build_external_dependency_calls(
        sample_repo.gateway,
        ExternalDependenciesStepConfig(snapshot=sample_repo.snapshot),
        inputs=ExternalDependencyInputs(
            catalog_provider=catalog,
            module_map=sample_repo.module_map,
            ast_by_goid=sample_repo.ast_map,
            features_map=sample_repo.features,
        ),
    )
    assert count_table_rows(sample_repo, "analytics.external_dependency_calls") >= 1

    build_entrypoints(
        sample_repo.gateway,
        EntryPointsStepConfig(snapshot=sample_repo.snapshot),
        catalog_provider=catalog,
        module_map=sample_repo.module_map,
        features_map=sample_repo.features,
    )
    assert count_table_rows(sample_repo, "analytics.entrypoints") >= 1
