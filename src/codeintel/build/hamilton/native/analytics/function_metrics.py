"""Native Hamilton implementation for function_metrics target.

This module provides the Hamilton native nodes for function metrics computation
with DAG-visible I/O via SaveToDecorator/DuckDBRowsSaver:

- `t__function_metrics__compute`: Pure compute node returning FunctionAnalyticsResult
- `function_metrics__metrics_rows`: Extract metrics rows for materialization
- `function_metrics__types_rows`: Extract types rows for materialization
- `function_metrics__validation_rows`: Extract validation rows for materialization
- `t__function_metrics`: Materialize node combining all table writes

The compute node calls `compute_function_analytics_result` which returns pure rows
without persistence. Persistence is handled by DuckDBRowsSaver via SaveToDecorator,
making I/O visible in the Hamilton DAG for caching and observability.

Phase 4: Analytics domain migration with Hamilton-native DAG-visible I/O.
"""

from __future__ import annotations

from codeintel.analytics.functions import FunctionAnalyticsOptions
from codeintel.analytics.functions.metrics import (
    FunctionAnalyticsResult,
    compute_function_analytics_result,
)
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
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
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
)
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.schemas.row_serialization import row_to_tuple

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    InputHashOptions,
    TargetGraph,
    TargetRunRecord,
    FunctionAnalyticsResult,
)

FUNCTION_METRICS_TARGET_NAME = "function_metrics"

FUNCTION_METRICS_TABLE_KEY = "analytics.function_metrics"
FUNCTION_TYPES_TABLE_KEY = "analytics.function_types"
FUNCTION_VALIDATION_TABLE_KEY = "analytics.function_validation"
FUNCTION_METRICS_TABLE_KEYS = (
    FUNCTION_METRICS_TABLE_KEY,
    FUNCTION_TYPES_TABLE_KEY,
    FUNCTION_VALIDATION_TABLE_KEY,
)
FUNCTION_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
    hash_options_node="function_metrics__hash_options",
)


@tag_helper(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def function_metrics__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for function_metrics execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, FUNCTION_METRICS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def function_metrics__skip(
    env: BuildEnv,
    graph: TargetGraph,
    function_metrics__hash_options: InputHashOptions,
) -> bool:
    """Return True when function_metrics should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        FUNCTION_METRICS_TARGET_NAME,
        hash_options=function_metrics__hash_options,
    )
    return executor.should_skip()


@tag_compute(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def t__function_metrics__compute(
    env: BuildEnv,
    *,
    function_metrics__skip: bool,
) -> FunctionAnalyticsResult | None:
    """Compute function metrics and type coverage for all functions.

    This is a pure compute node that returns rows without persistence.
    The actual DB writes are handled by downstream SaveToDecorator nodes.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    function_metrics__skip
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    FunctionAnalyticsResult | None
        Result containing metrics and types rows, or None if skipped.

    Notes
    -----
    The metrics computed include:
    - Lines of code (LOC, SLOC)
    - Cyclomatic complexity
    - Nesting depth
    - Type annotation coverage
    """
    if function_metrics__skip:
        return None

    options = load_target_options(
        env,
        target_name=FUNCTION_METRICS_TARGET_NAME,
        options_type=FunctionAnalyticsOptions,
    )
    return compute_function_analytics_result(
        env.gateway,
        env.snapshot,
        options=options,
    )


@save_rows(
    context=FUNCTION_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FUNCTION_METRICS_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
    target_="function_metrics__metrics_rows",
)
def function_metrics__metrics_rows(
    t__function_metrics__compute: FunctionAnalyticsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract metrics rows for analytics.function_metrics table.

    Parameters
    ----------
    t__function_metrics__compute
        Computed function analytics result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.function_metrics table, or None if compute
        result is None.
    """
    if t__function_metrics__compute is None:
        return None
    return tuple(
        row_to_tuple(FUNCTION_METRICS_TABLE_KEY, row)
        for row in t__function_metrics__compute.metrics_rows
    )


@save_rows(
    context=FUNCTION_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FUNCTION_TYPES_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
    target_="function_metrics__types_rows",
)
def function_metrics__types_rows(
    t__function_metrics__compute: FunctionAnalyticsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract types rows for analytics.function_types table.

    Parameters
    ----------
    t__function_metrics__compute
        Computed function analytics result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.function_types table, or None if compute
        result is None.
    """
    if t__function_metrics__compute is None:
        return None
    return tuple(
        row_to_tuple(FUNCTION_TYPES_TABLE_KEY, row)
        for row in t__function_metrics__compute.types_rows
    )


@save_rows(
    context=FUNCTION_METRICS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=FUNCTION_VALIDATION_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
    target_="function_metrics__validation_rows",
)
def function_metrics__validation_rows(
    t__function_metrics__compute: FunctionAnalyticsResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract validation rows for analytics.function_validation table.

    Parameters
    ----------
    t__function_metrics__compute
        Computed function analytics result from the compute node.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Rows for the analytics.function_validation table, or None if compute
        result is None or no validation issues.
    """
    if t__function_metrics__compute is None:
        return None
    validation_rows = t__function_metrics__compute.reporter.to_rows()
    if not validation_rows:
        return None
    return tuple(validation_rows)


function_metrics__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=FUNCTION_METRICS_TARGET_NAME,
    table_keys=FUNCTION_METRICS_TABLE_KEYS,
    node_name="function_metrics__table_materializations",
)


@codeintel_target(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    function_metrics__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Materialize function structural metrics and type annotations.

    Combines materialization metadata from all three table writes into a
    single TargetRunRecord for the function_metrics target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    function_metrics__table_materializations
        Aggregated materialization metadata for function_metrics tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            graph=graph,
            target_name=FUNCTION_METRICS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=function_metrics__table_materializations,
    )


__all__ = [
    "FunctionAnalyticsResult",
    "function_metrics__hash_options",
    "function_metrics__metrics_rows",
    "function_metrics__skip",
    "function_metrics__table_materializations",
    "function_metrics__types_rows",
    "function_metrics__validation_rows",
    "t__function_metrics",
    "t__function_metrics__compute",
]
