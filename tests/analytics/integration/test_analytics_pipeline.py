"""Integration-style tests for analytics compute modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.data_models.core import compute_data_models
from codeintel.analytics.dependencies.core import (
    ExternalDependencyInputs,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints.core import EntrypointBuildInputs, build_entrypoints
from codeintel.analytics.functions.function_contracts import compute_function_contracts
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    compute_function_effects,
)
from codeintel.analytics.functions.function_history import compute_function_history
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.core.catalog import CatalogService
from tests._helpers.assertions import expect_true
from tests.analytics.integration.sample_repo import (
    build_runtime,
    count_table_rows,
)

if TYPE_CHECKING:
    from tests.analytics.integration.sample_repo import (
        SampleRepo,
    )

MIN_ROWS = 2


def test_full_analytics_pipeline(sample_repo: SampleRepo) -> None:
    """Execute compute flows end-to-end on a small in-memory snapshot."""
    catalog = CatalogService.from_db(
        sample_repo.gateway,
        repo=sample_repo.snapshot.repo,
        commit=sample_repo.snapshot.commit,
    )

    summary = compute_function_metrics_and_types(
        sample_repo.gateway,
        sample_repo.snapshot,
    )
    expect_true(summary["metrics_rows"] >= MIN_ROWS)

    compute_function_contracts(
        sample_repo.gateway,
        sample_repo.snapshot,
        function_ast_map=sample_repo.ast_map,
        catalog=catalog,
        max_conditions_per_func=5,
    )
    expect_true(count_table_rows(sample_repo, "analytics.function_contracts") >= MIN_ROWS)

    runtime = build_runtime(sample_repo)
    compute_function_effects(
        sample_repo.gateway,
        sample_repo.snapshot,
        inputs=FunctionEffectsInputs(
            catalog_provider=catalog,
            runtime=runtime,
            ast_map=sample_repo.ast_map,
            missing_goids=set(),
        ),
    )
    expect_true(count_table_rows(sample_repo, "analytics.function_effects") >= MIN_ROWS)

    compute_function_history(
        sample_repo.gateway,
        sample_repo.snapshot,
    )
    expect_true(count_table_rows(sample_repo, "analytics.function_history") >= MIN_ROWS)

    compute_data_models(
        sample_repo.gateway,
        sample_repo.snapshot,
    )
    expect_true(count_table_rows(sample_repo, "analytics.data_models") >= 1)

    build_external_dependency_calls(
        sample_repo.gateway,
        sample_repo.snapshot,
        inputs=ExternalDependencyInputs(
            catalog_provider=catalog,
            module_map=sample_repo.module_map,
            ast_by_goid=sample_repo.ast_map,
            features_map=sample_repo.features,
        ),
    )
    expect_true(count_table_rows(sample_repo, "analytics.external_dependency_calls") >= 1)

    inputs = EntrypointBuildInputs(
        catalog_provider=catalog,
        module_map=sample_repo.module_map,
        features_map=sample_repo.features,
    )
    build_entrypoints(
        sample_repo.gateway,
        sample_repo.snapshot,
        inputs,
    )
    expect_true(count_table_rows(sample_repo, "analytics.entrypoints") >= 1)
