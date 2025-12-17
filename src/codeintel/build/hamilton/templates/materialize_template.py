"""Unified materialization helpers for Hamilton build targets.

This module consolidates the common glue across the Phase 1 templates:

- NativeTargetExecutor-based "executor" pattern
- DuckDB row-oriented materialization (rows → DuckDBRowsSaver → TargetRunRecord)
- DuckDB Ibis materialization (Ibis expr → DuckDBIbisTableSaver → TargetRunRecord)

It is designed to be used directly via `@subdag` wiring in native target modules, or via
re-exports from `codeintel.build.hamilton.templates`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ibis.expr.types as ir
from hamilton.function_modifiers import source, tag
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver, DuckDBRowsSaver
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.targets import TargetGraph

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Keep types available for Hamilton's runtime type resolution
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)

# Note: Using Any instead of Protocol for compute_result parameter because:
# 1. Python 3.13 Protocol doesn't support issubclass() for data-only protocols
# 2. Hamilton internally uses issubclass() for type matching
# The expected interface is:
#   - success: bool
#   - table_counts: dict[str, int]
#   - error: str | None
ComputeResult = Any


@tag(node_type="materialize")
def executor_materialize(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ComputeResult,
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

    def compute() -> dict[str, int]:
        return dict(compute_result.table_counts)

    return executor.execute(compute)


@tag(node_type="materialize")
def executor_record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ComputeResult,
) -> TargetRunRecord:
    """Short-name wrapper for executor_materialize for subDAG wiring.

    Returns
    -------
    TargetRunRecord
        Execution record for the target.
    """
    return executor_materialize(env, graph, target_name, compute_result)


@SaveToDecorator(
    [DuckDBRowsSaver],
    output_name_="materialization",
    env=source("env"),
    graph=source("graph"),
    target_name=source("target_name"),
    table_key=source("table_key"),
    columns=source("columns"),
)
@tag(node_type="compute")
def rows_to_save(
    rows: Sequence[tuple[Any, ...]] | None,
) -> Sequence[tuple[Any, ...]] | None:
    """Return the row sequence to be materialized.

    Returns
    -------
    Sequence[tuple[Any, ...]] | None
        The same row sequence.
    """
    return rows


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_="materialization",
    env=source("env"),
    graph=source("graph"),
    target_name=source("target_name"),
    table_key=source("table_key"),
)
@tag(node_type="compute")
def ibis_expr_to_save(expr: ir.Table | None) -> ir.Table | None:
    """Return the Ibis expression to be materialized.

    Returns
    -------
    ir.Table | None
        The same Ibis table expression.
    """
    return expr


@tag(node_type="materialize")
def duckdb_record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    table_key: str,
    materialization: dict[str, Any],
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


def row_to_tuple(row: Mapping[str, object], columns: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a mapping row to a tuple in column order.

    Returns
    -------
    tuple[object, ...]
        Tuple of row values in column order.
    """
    return tuple(row.get(col) for col in columns)


def rows_to_tuples(
    rows: Sequence[Mapping[str, object]],
    columns: tuple[str, ...],
) -> tuple[tuple[object, ...], ...]:
    """Convert a sequence of mapping rows to a tuple of tuples in column order.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Tuple of row tuples in column order.
    """
    return tuple(row_to_tuple(row, columns) for row in rows)


__all__ = [
    "ComputeResult",
    "duckdb_record",
    "executor_materialize",
    "executor_record",
    "ibis_expr_to_save",
    "row_to_tuple",
    "rows_to_save",
    "rows_to_tuples",
]
