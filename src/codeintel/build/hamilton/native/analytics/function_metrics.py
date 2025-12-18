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

import logging
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_analytics_result,
)
from codeintel.analytics.functions.metrics import FunctionAnalyticsResult
from codeintel.analytics.parsing.validation import FUNCTION_VALIDATION_COLS
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, FunctionAnalyticsResult)

FUNCTION_METRICS_TARGET_NAME = "function_metrics"

FUNCTION_METRICS_TABLE_KEY = "analytics.function_metrics"
FUNCTION_TYPES_TABLE_KEY = "analytics.function_types"
FUNCTION_VALIDATION_TABLE_KEY = "analytics.function_validation"
FUNCTION_METRICS_TABLE_KEYS = (
    FUNCTION_METRICS_TABLE_KEY,
    FUNCTION_TYPES_TABLE_KEY,
    FUNCTION_VALIDATION_TABLE_KEY,
)

TARGET_SPECS = (
    make_output_target(
        name=FUNCTION_METRICS_TARGET_NAME,
        module="analytics",
        description="Function structural metrics and type annotations.",
        options=TargetSpecOptions(
            table_keys=FUNCTION_METRICS_TABLE_KEYS,
        ),
    ),
)

# Column definitions for function_metrics table
FUNCTION_METRICS_COLS: tuple[str, ...] = (
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "loc",
    "logical_loc",
    "param_count",
    "positional_params",
    "keyword_only_params",
    "has_varargs",
    "has_varkw",
    "is_async",
    "is_generator",
    "return_count",
    "yield_count",
    "raise_count",
    "cyclomatic_complexity",
    "max_nesting_depth",
    "stmt_count",
    "decorator_count",
    "has_docstring",
    "complexity_bucket",
    "created_at",
)

# Column definitions for function_types table
FUNCTION_TYPES_COLS: tuple[str, ...] = (
    "function_goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "total_params",
    "annotated_params",
    "unannotated_params",
    "param_typed_ratio",
    "has_return_annotation",
    "return_type",
    "return_type_source",
    "type_comment",
    "param_types",
    "fully_typed",
    "partial_typed",
    "untyped",
    "typedness_bucket",
    "typedness_source",
    "created_at",
)


def _row_to_tuple(row: Mapping[str, object], cols: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a TypedDict row to a tuple in column order.

    Parameters
    ----------
    row
        Row mapping from column name to value.
    cols
        Column names in the desired order.

    Returns
    -------
    tuple[object, ...]
        Values in column order.
    """
    return tuple(row.get(col) for col in cols)


@tag_compute(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def t__function_metrics__compute(
    env: BuildEnv, graph: TargetGraph
) -> FunctionAnalyticsResult | None:
    """Compute function metrics and type coverage for all functions.

    This is a pure compute node that returns rows without persistence.
    The actual DB writes are handled by downstream SaveToDecorator nodes.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for skip detection.

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
    target = graph.get(FUNCTION_METRICS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, FUNCTION_METRICS_TARGET_NAME)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=options_hash,
            manifests=env.manifest_index,
        )
        if should_skip_native_target(env, target, input_hash):
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_METRICS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_METRICS_TARGET_NAME),
    table_key=value(FUNCTION_METRICS_TABLE_KEY),
    columns=value(FUNCTION_METRICS_COLS),
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
        _row_to_tuple(row, FUNCTION_METRICS_COLS)
        for row in t__function_metrics__compute.metrics_rows
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_TYPES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_METRICS_TARGET_NAME),
    table_key=value(FUNCTION_TYPES_TABLE_KEY),
    columns=value(FUNCTION_TYPES_COLS),
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
        _row_to_tuple(row, FUNCTION_TYPES_COLS) for row in t__function_metrics__compute.types_rows
    )


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(FUNCTION_VALIDATION_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(FUNCTION_METRICS_TARGET_NAME),
    table_key=value(FUNCTION_VALIDATION_TABLE_KEY),
    columns=value(tuple(FUNCTION_VALIDATION_COLS)),
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


@tag_materialize(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
def t__function_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__function_metrics: MaterializationMetadata,
    m__analytics__function_types: MaterializationMetadata,
    m__analytics__function_validation: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize function metrics target.

    Combines materialization metadata from all three table writes into a
    single TargetRunRecord for the function_metrics target.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup.
    m__analytics__function_metrics
        Materialization metadata for function_metrics table.
    m__analytics__function_types
        Materialization metadata for function_types table.
    m__analytics__function_validation
        Materialization metadata for function_validation table.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=FUNCTION_METRICS_TARGET_NAME,
        materializations={
            FUNCTION_METRICS_TABLE_KEY: m__analytics__function_metrics,
            FUNCTION_TYPES_TABLE_KEY: m__analytics__function_types,
            FUNCTION_VALIDATION_TABLE_KEY: m__analytics__function_validation,
        },
    )


__all__ = [
    "FUNCTION_METRICS_COLS",
    "FUNCTION_TYPES_COLS",
    "FunctionAnalyticsResult",
    "function_metrics__metrics_rows",
    "function_metrics__types_rows",
    "function_metrics__validation_rows",
    "t__function_metrics",
    "t__function_metrics__compute",
]
