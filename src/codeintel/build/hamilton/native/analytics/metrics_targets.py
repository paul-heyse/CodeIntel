"""Consolidated Hamilton implementation for metrics-related analytics targets.

This module consolidates metrics analytics targets using native materialization helpers:

History Targets (Pattern B - Rows):
- ``function_history``: Per-function creation/modification/churn metrics
- ``history_timeseries``: Multi-commit timeseries analytics

Graph Metrics Targets (Rows):
- ``subsystem_graph_metrics``: Metrics for subsystem coupling and centrality
- ``symbol_graph_metrics``: Metrics for symbol usage patterns
- ``subsystem_agreement``: Subsystem assignment agreement metrics

Multi-Table Target (Pattern C):
- ``test_graph_metrics``: Metrics from test-function bipartite graph

Graph metrics targets use DAG-visible row materialization via ``save_rows``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.functions.function_history import build_function_history_rows
from codeintel.analytics.graphs.subsystem_agreement import (
    build_subsystem_agreement_rows,
)
from codeintel.analytics.graphs.subsystem_graph_metrics import (
    build_subsystem_graph_metrics_rows,
)
from codeintel.analytics.graphs.symbol_graph_metrics import (
    build_symbol_graph_metrics_function_rows,
    build_symbol_graph_metrics_module_rows,
)
from codeintel.analytics.history.history_timeseries import build_history_timeseries_rows
from codeintel.analytics.testing.compute import (
    TestGraphMetricsResult,
    compute_test_graph_metrics_pure,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.graph_runtime_options import load_graph_runtime_options
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    SaverContext,
    TableSaveSpec,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
)
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.graphs.runtime import GraphRuntime, resolve_graph_runtime
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    GraphRuntime,
    DagCatalog,
    TargetRunRecord,
    TestGraphMetricsResult,
)

if TYPE_CHECKING:
    from codeintel.analytics.compute.row_builders import SubsystemMetricRow

FUNCTION_HISTORY_TARGET_NAME = "function_history"
HISTORY_TIMESERIES_TARGET_NAME = "history_timeseries"
SUBSYSTEM_GRAPH_METRICS_TARGET_NAME = "subsystem_graph_metrics"
SYMBOL_GRAPH_METRICS_TARGET_NAME = "symbol_graph_metrics"
SUBSYSTEM_AGREEMENT_TARGET_NAME = "subsystem_agreement"
TEST_GRAPH_METRICS_TARGET_NAME = "test_graph_metrics"

FUNCTION_HISTORY_TABLE_KEY = "analytics.function_history"
HISTORY_TIMESERIES_TABLE_KEY = "analytics.history_timeseries"
SUBSYSTEM_GRAPH_METRICS_TABLE_KEY = "analytics.subsystem_graph_metrics"
SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY = "analytics.symbol_graph_metrics_modules"
SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.symbol_graph_metrics_functions"
SYMBOL_GRAPH_METRICS_TABLE_KEYS = (
    SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY,
    SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
)
SUBSYSTEM_AGREEMENT_TABLE_KEY = "analytics.subsystem_agreement"
TEST_GRAPH_METRICS_TESTS_TABLE_KEY = "analytics.test_graph_metrics_tests"
TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY = "analytics.test_graph_metrics_functions"
TEST_GRAPH_METRICS_TABLE_KEYS = (
    TEST_GRAPH_METRICS_TESTS_TABLE_KEY,
    TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY,
)
FUNCTION_HISTORY_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_HISTORY_TARGET_NAME,
)
HISTORY_TIMESERIES_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
)
SUBSYSTEM_GRAPH_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
)
SYMBOL_GRAPH_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
)
SUBSYSTEM_AGREEMENT_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
)
TEST_GRAPH_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=TEST_GRAPH_METRICS_TARGET_NAME,
)


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for metrics nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


























def _get_graph_runtime(
    env: BuildEnv,
    *,
    target_name: str,
    gateway: StorageGateway,
) -> GraphRuntime | None:
    """Resolve graph runtime from build environment.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    target_name
        Target name used to load runtime options from configuration.
    gateway
        Storage gateway used to resolve runtime metadata.

    Returns
    -------
    GraphRuntime | None
        Resolved graph runtime, or None if resolution fails.
    """
    try:
        options = load_graph_runtime_options(env, target_name=target_name)
        return resolve_graph_runtime(gateway, env.snapshot, options)
    except (RuntimeError, ValueError) as exc:
        log.warning("Failed to resolve graph runtime: %s", exc)
        return None


# -----------------------------------------------------------------------------
# Function history (Pattern B - Rows)
# -----------------------------------------------------------------------------


@save_rows(
    context=FUNCTION_HISTORY_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FUNCTION_HISTORY_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_HISTORY_TARGET_NAME,
    target_="t__function_history__compute",
)
def t__function_history__compute(
    env: BuildEnv,
    gateway: StorageGateway,
) -> tuple[tuple[object, ...], ...] | None:
    """Compute function history metrics for all functions.

    This is a pure compute node with no side effects. It computes git history
    and churn metrics for each function and returns row data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples matching the function_history schema order, or None when skipped.

    Notes
    -----
    The metrics computed include:
    - Function creation and last modification dates
    - Commit count and author count
    - Lines added and deleted (churn)
    - Stability bucket classification
    """

    rows = build_function_history_rows(
        gateway,
        env.snapshot,
        runner=env.providers.tool_runner,
    )
    return tuple(rows)


@codeintel_target(domain="analytics", target=FUNCTION_HISTORY_TARGET_NAME)
def t__function_history(
    env: BuildEnv,
    catalog: DagCatalog,
    function_history__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize function git history and churn metrics.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    function_history__table_materializations
        Materialization results for analytics.function_history.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=FUNCTION_HISTORY_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=function_history__table_materializations,
    )


function_history__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=FUNCTION_HISTORY_TARGET_NAME,
    table_keys=(FUNCTION_HISTORY_TABLE_KEY,),
)


# -----------------------------------------------------------------------------
# History timeseries (Pattern B - Rows)
# -----------------------------------------------------------------------------


@save_rows(
    context=HISTORY_TIMESERIES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=HISTORY_TIMESERIES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
    target_="t__history_timeseries__compute",
)
def t__history_timeseries__compute(
    env: BuildEnv,
) -> tuple[tuple[object, ...], ...] | None:
    """Compute history timeseries metrics across commits.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples matching the history_timeseries schema, or None when skipped.
    """

    options = env.history_options
    gateway_resolver = env.history_db_resolver
    if options is None or gateway_resolver is None:
        log.info(
            "history_timeseries: missing history options or gateway resolver; "
            "returning empty result set."
        )
        return ()
    rows = build_history_timeseries_rows(
        env.snapshot,
        gateway_resolver,
        options=options,
        runner=env.providers.tool_runner,
    )
    return tuple(rows)


@codeintel_target(domain="analytics", target=HISTORY_TIMESERIES_TARGET_NAME)
def t__history_timeseries(
    env: BuildEnv,
    catalog: DagCatalog,
    history_timeseries__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize historical metrics timeseries for trending.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    history_timeseries__table_materializations
        Materialization results for analytics.history_timeseries.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=HISTORY_TIMESERIES_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=history_timeseries__table_materializations,
    )


history_timeseries__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=HISTORY_TIMESERIES_TARGET_NAME,
    table_keys=(HISTORY_TIMESERIES_TABLE_KEY,),
)


# -----------------------------------------------------------------------------
# Graph metrics (Rows)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsystemGraphMetricsComputeResult:
    """Result from subsystem graph metrics computation."""

    rows: list[SubsystemMetricRow] | None
    error: str | None = None


@dataclass(frozen=True)
class SymbolGraphMetricsComputeResult:
    """Result from symbol graph metrics computation."""

    module_rows: list[tuple[object, ...]] | None
    function_rows: list[tuple[object, ...]] | None
    error: str | None = None


@dataclass(frozen=True)
class SubsystemAgreementComputeResult:
    """Result from subsystem agreement computation."""

    rows: list[tuple[object, ...]] | None
    error: str | None = None


@tag_compute(domain="analytics", target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
def t__subsystem_graph_metrics__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__subsystems: TargetRunRecord,
) -> SubsystemGraphMetricsComputeResult:
    """Compute graph metrics for subsystems.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__subsystems
        Upstream subsystems target result (for dependency).
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    SubsystemGraphMetricsComputeResult
        Row results and optional error.
    """
    if t__subsystems.status != "succeeded":
        return SubsystemGraphMetricsComputeResult(
            rows=None,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )


    try:
        graph_runtime = _get_graph_runtime(
            env,
            target_name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
            gateway=gateway,
        )
        log.info(
            "Computing subsystem graph metrics for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        rows = build_subsystem_graph_metrics_rows(
            gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            runtime=graph_runtime,
        )
        return SubsystemGraphMetricsComputeResult(rows=rows)

    except Exception as exc:
        log.exception("Subsystem graph metrics computation failed")
        return SubsystemGraphMetricsComputeResult(rows=None, error=str(exc))


@save_rows(
    context=SUBSYSTEM_GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    target_="subsystem_graph_metrics__rows",
)
def subsystem_graph_metrics__rows(
    t__subsystem_graph_metrics__compute: SubsystemGraphMetricsComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.subsystem_graph_metrics table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__subsystem_graph_metrics__compute.rows is None:
        return None
    return tuple(t__subsystem_graph_metrics__compute.rows)


@codeintel_target(domain="analytics", target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
def t__subsystem_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    t__subsystem_graph_metrics__compute: SubsystemGraphMetricsComputeResult,
    subsystem_graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize graph metrics for subsystems.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    t__subsystem_graph_metrics__compute
        Computed subsystem graph metrics result.
    subsystem_graph_metrics__table_materializations
        Materialization results for analytics.subsystem_graph_metrics.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__subsystem_graph_metrics__compute.error:
        options_hash = options_hash_for_target(env, SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
        return TargetRunRecord(
            target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__subsystem_graph_metrics__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=subsystem_graph_metrics__table_materializations,
    )


subsystem_graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    table_keys=(SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,),
)


# -----------------------------------------------------------------------------
# Symbol graph metrics (Rows)
# -----------------------------------------------------------------------------


@tag_compute(domain="analytics", target=SYMBOL_GRAPH_METRICS_TARGET_NAME)
def t__symbol_graph_metrics__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__symbol_uses: TargetRunRecord,
) -> SymbolGraphMetricsComputeResult:
    """Compute graph metrics from symbol usage patterns.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__symbol_uses
        Upstream symbol_uses target result (for dependency).
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    SymbolGraphMetricsComputeResult
        Row results and optional error.
    """
    if t__symbol_uses.status != "succeeded":
        return SymbolGraphMetricsComputeResult(
            module_rows=None,
            function_rows=None,
            error=f"Upstream symbol_uses target failed: {t__symbol_uses.error}",
        )


    module_rows: list[tuple[object, ...]] | None = None
    function_rows: list[tuple[object, ...]] | None = None
    errors: list[str] = []

    try:
        graph_runtime = _get_graph_runtime(
            env,
            target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
            gateway=gateway,
        )
        repo = env.snapshot.repo
        commit = env.snapshot.commit

        try:
            log.info("Computing symbol graph metrics (modules) for %s@%s", repo, commit)
            module_rows = build_symbol_graph_metrics_module_rows(
                gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            errors.append(f"modules: {exc}")
            log.warning("Symbol graph metrics (modules) failed: %s", exc)

        try:
            log.info("Computing symbol graph metrics (functions) for %s@%s", repo, commit)
            function_rows = build_symbol_graph_metrics_function_rows(
                gateway,
                repo=repo,
                commit=commit,
                runtime=graph_runtime,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            errors.append(f"functions: {exc}")
            log.warning("Symbol graph metrics (functions) failed: %s", exc)

        if errors:
            return SymbolGraphMetricsComputeResult(
                module_rows=module_rows,
                function_rows=function_rows,
                error="; ".join(errors),
            )
        return SymbolGraphMetricsComputeResult(
            module_rows=module_rows,
            function_rows=function_rows,
        )

    except Exception as exc:
        log.exception("Symbol graph metrics computation failed")
        return SymbolGraphMetricsComputeResult(
            module_rows=None,
            function_rows=None,
            error=str(exc),
        )


@save_rows(
    context=SYMBOL_GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SYMBOL_GRAPH_METRICS_MODULES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    target_="symbol_graph_metrics__modules_rows",
)
def symbol_graph_metrics__modules_rows(
    t__symbol_graph_metrics__compute: SymbolGraphMetricsComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.symbol_graph_metrics_modules.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__symbol_graph_metrics__compute.error:
        return None
    if t__symbol_graph_metrics__compute.module_rows is None:
        return None
    return tuple(t__symbol_graph_metrics__compute.module_rows)


@save_rows(
    context=SYMBOL_GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SYMBOL_GRAPH_METRICS_FUNCTIONS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    target_="symbol_graph_metrics__functions_rows",
)
def symbol_graph_metrics__functions_rows(
    t__symbol_graph_metrics__compute: SymbolGraphMetricsComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.symbol_graph_metrics_functions.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__symbol_graph_metrics__compute.error:
        return None
    if t__symbol_graph_metrics__compute.function_rows is None:
        return None
    return tuple(t__symbol_graph_metrics__compute.function_rows)


@codeintel_target(domain="analytics", target=SYMBOL_GRAPH_METRICS_TARGET_NAME)
def t__symbol_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    t__symbol_graph_metrics__compute: SymbolGraphMetricsComputeResult,
    symbol_graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize graph metrics from symbol usage patterns.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    t__symbol_graph_metrics__compute
        Computed symbol graph metrics result.
    symbol_graph_metrics__table_materializations
        Materialization results for analytics.symbol_graph_metrics tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__symbol_graph_metrics__compute.error:
        options_hash = options_hash_for_target(env, SYMBOL_GRAPH_METRICS_TARGET_NAME)
        return TargetRunRecord(
            target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__symbol_graph_metrics__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SYMBOL_GRAPH_METRICS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=symbol_graph_metrics__table_materializations,
    )


symbol_graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SYMBOL_GRAPH_METRICS_TARGET_NAME,
    table_keys=SYMBOL_GRAPH_METRICS_TABLE_KEYS,
)


# -----------------------------------------------------------------------------
# Subsystem agreement (Rows)
# -----------------------------------------------------------------------------


@tag_compute(domain="analytics", target=SUBSYSTEM_AGREEMENT_TARGET_NAME)
def t__subsystem_agreement__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__subsystems: TargetRunRecord,
) -> SubsystemAgreementComputeResult:
    """Compare subsystem assignments with import community labels.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__subsystems
        Upstream subsystems target result (for dependency).
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    SubsystemAgreementComputeResult
        Row results and optional error message.
    """
    if t__subsystems.status != "succeeded":
        return SubsystemAgreementComputeResult(
            rows=None,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )


    try:
        log.info(
            "Computing subsystem agreement for %s@%s",
            env.snapshot.repo,
            env.snapshot.commit,
        )
        rows = build_subsystem_agreement_rows(
            gateway,
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        return SubsystemAgreementComputeResult(rows=rows)
    except Exception as exc:
        log.exception("Subsystem agreement computation failed")
        return SubsystemAgreementComputeResult(rows=None, error=str(exc))


@save_rows(
    context=SUBSYSTEM_AGREEMENT_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    target_="subsystem_agreement__rows",
)
def subsystem_agreement__rows(
    t__subsystem_agreement__compute: SubsystemAgreementComputeResult,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.subsystem_agreement table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if t__subsystem_agreement__compute.rows is None:
        return None
    return tuple(t__subsystem_agreement__compute.rows)


@codeintel_target(domain="analytics", target=SUBSYSTEM_AGREEMENT_TARGET_NAME)
def t__subsystem_agreement(
    env: BuildEnv,
    catalog: DagCatalog,
    t__subsystem_agreement__compute: SubsystemAgreementComputeResult,
    subsystem_agreement__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Compute subsystem vs import community agreement.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    t__subsystem_agreement__compute
        Computed subsystem agreement result.
    subsystem_agreement__table_materializations
        Materialization results for analytics.subsystem_agreement.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    if t__subsystem_agreement__compute.error:
        options_hash = options_hash_for_target(env, SUBSYSTEM_AGREEMENT_TARGET_NAME)
        return TargetRunRecord(
            target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__subsystem_agreement__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SUBSYSTEM_AGREEMENT_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=subsystem_agreement__table_materializations,
    )


subsystem_agreement__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    table_keys=(SUBSYSTEM_AGREEMENT_TABLE_KEY,),
)


# -----------------------------------------------------------------------------
# Test graph metrics (Pattern C - Multi-Table)
# -----------------------------------------------------------------------------


@tag_tool(domain="analytics", target=TEST_GRAPH_METRICS_TARGET_NAME)
def t__test_graph_metrics__compute(
    env: BuildEnv,
    gateway: StorageGateway,
) -> TestGraphMetricsResult | None:
    """Compute test graph metrics for all tests and functions.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    TestGraphMetricsResult | None
        Container with rows for test and function metrics tables, or None if skipped.
    """

    runtime = _get_graph_runtime(
        env,
        target_name=TEST_GRAPH_METRICS_TARGET_NAME,
        gateway=gateway,
    )
    if runtime is None:
        return TestGraphMetricsResult(test_rows=(), function_rows=())
    return compute_test_graph_metrics_pure(gateway, env.snapshot, runtime)


@save_rows(
    context=TEST_GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=TEST_GRAPH_METRICS_TESTS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=TEST_GRAPH_METRICS_TARGET_NAME,
    target_="test_graph_metrics__tests_rows",
)
def test_graph_metrics__tests_rows(
    t__test_graph_metrics__compute: TestGraphMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_graph_metrics_tests.

    Parameters
    ----------
    t__test_graph_metrics__compute
        Computed test graph metrics result.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for the test metrics table, or None if compute was skipped.
    """
    if t__test_graph_metrics__compute is None:
        return None
    return tuple(t__test_graph_metrics__compute.test_rows)


@save_rows(
    context=TEST_GRAPH_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=TEST_GRAPH_METRICS_FUNCTIONS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=TEST_GRAPH_METRICS_TARGET_NAME,
    target_="test_graph_metrics__functions_rows",
)
def test_graph_metrics__functions_rows(
    t__test_graph_metrics__compute: TestGraphMetricsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.test_graph_metrics_functions.

    Parameters
    ----------
    t__test_graph_metrics__compute
        Computed test graph metrics result.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples for the function metrics table, or None if compute was skipped.
    """
    if t__test_graph_metrics__compute is None:
        return None
    return tuple(t__test_graph_metrics__compute.function_rows)


@codeintel_target(domain="analytics", target=TEST_GRAPH_METRICS_TARGET_NAME)
def t__test_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    test_graph_metrics__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize graph metrics from test-function bipartite graph.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup.
    test_graph_metrics__table_materializations
        Materialization results for test graph metrics tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=TEST_GRAPH_METRICS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=test_graph_metrics__table_materializations,
    )


test_graph_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=TEST_GRAPH_METRICS_TARGET_NAME,
    table_keys=TEST_GRAPH_METRICS_TABLE_KEYS,
)


__all__ = [
    "SubsystemAgreementComputeResult",
    "SubsystemGraphMetricsComputeResult",
    "SymbolGraphMetricsComputeResult",
    "function_history__table_materializations",
    "history_timeseries__table_materializations",
    "subsystem_agreement__rows",
    "subsystem_agreement__table_materializations",
    "subsystem_graph_metrics__rows",
    "subsystem_graph_metrics__table_materializations",
    "symbol_graph_metrics__functions_rows",
    "symbol_graph_metrics__modules_rows",
    "symbol_graph_metrics__table_materializations",
    "t__function_history",
    "t__function_history__compute",
    "t__history_timeseries",
    "t__history_timeseries__compute",
    "t__subsystem_agreement",
    "t__subsystem_agreement__compute",
    "t__subsystem_graph_metrics",
    "t__subsystem_graph_metrics__compute",
    "t__symbol_graph_metrics",
    "t__symbol_graph_metrics__compute",
    "t__test_graph_metrics",
    "t__test_graph_metrics__compute",
    "test_graph_metrics__functions_rows",
    "test_graph_metrics__table_materializations",
    "test_graph_metrics__tests_rows",
]
