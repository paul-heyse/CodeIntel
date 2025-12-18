"""Unified materialization helpers for Hamilton build targets.

This module consolidates the common glue across the Phase 1 templates:

- NativeTargetExecutor-based "executor" pattern
- DuckDB Ibis materialization (Ibis expr → DuckDBIbisTableSaver → TargetRunRecord)

It is designed to be used directly via `@subdag` wiring in native target modules, or via
re-exports from `codeintel.build.hamilton.templates`.
"""

from __future__ import annotations

import ibis.expr.types as ir
from hamilton.function_modifiers import source

from codeintel.build.hamilton.boundary_types import MaterializationMetadata, RowCounts
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.targets import TargetGraph

# Keep types available for Hamilton's runtime type resolution
_HAMILTON_TYPE_HINTS = (BuildEnv, ExecutionResult, TargetGraph, TargetRunRecord, ir.Table)


@tag_materialize()
def executor_materialize(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ExecutionResult,
) -> TargetRunRecord:
    """Materialize using the NativeTargetExecutor pattern.

    Returns
    -------
    TargetRunRecord
        Execution record for the target.
    """
    executor = NativeTargetExecutor.for_target(env, graph, target_name)

    if executor.should_skip():
        return executor.skip()

    if not compute_result.success:
        error_msg = compute_result.error or f"{target_name} computation failed"
        return executor.fail(RuntimeError(error_msg))

    def compute() -> RowCounts:
        return dict(compute_result.table_counts)

    return executor.execute(compute)


@tag_materialize()
def executor_record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ExecutionResult,
) -> TargetRunRecord:
    """Short-name wrapper for executor_materialize for subDAG wiring.

    Returns
    -------
    TargetRunRecord
        Execution record for the target.
    """
    return executor_materialize(env, graph, target_name, compute_result)


@SaveToObjectMetadataDecorator(
    [DuckDBIbisTableSaver],
    output_name_="materialization",
    env=source("env"),
    graph=source("graph"),
    target_name=source("target_name"),
    table_key=source("table_key"),
)
@tag_compute()
def ibis_expr_to_save(expr: ir.Table | None) -> ir.Table | None:
    """Return the Ibis expression to be materialized.

    Returns
    -------
    ir.Table | None
        The same Ibis table expression.
    """
    return expr


@tag_materialize()
def duckdb_record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    table_key: str,
    materialization: MaterializationMetadata,
) -> TargetRunRecord:
    """Convert a saver metadata dict into a TargetRunRecord.

    Returns
    -------
    TargetRunRecord
        Execution record for the target.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name=target_name,
        expected_table_key=table_key,
        materialization=materialization,
    )


__all__ = [
    "duckdb_record",
    "executor_materialize",
    "executor_record",
    "ibis_expr_to_save",
]
