"""Unified materialization helpers for Hamilton build targets.

This module consolidates the common glue across the Phase 1 templates:

- NativeTargetExecutor-based "executor" pattern
- DuckDB Ibis materialization (Ibis expr → DuckDBIbisTableSaver → TargetRunRecord)

It is designed to be used directly via `@subdag` wiring in native target modules, or via
re-exports from `codeintel.build.hamilton.templates`.
"""

from __future__ import annotations

from types import ModuleType

import ibis.expr.types as ir
from hamilton.function_modifiers.dependencies import source, value

from codeintel.build.hamilton.boundary_types import MaterializationMetadata, RowCounts
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult, to_execution_result
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph

# Keep types available for Hamilton's runtime type resolution
_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    ExecutionResult,
    TargetGraph,
    TargetRunRecord,
    ir.Table,
)


@tag_materialize()
def executor_materialize(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ExecutionResult,
    *,
    hash_options: InputHashOptions | None = None,
) -> TargetRunRecord:
    """Materialize using the NativeTargetExecutor pattern.

    Returns
    -------
    TargetRunRecord
        Execution record for the target.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        target_name,
        hash_options=hash_options,
    )

    if executor.should_skip():
        return executor.skip()

    resolved = to_execution_result(
        compute_result,
        default_error=f"{target_name} computation failed",
    )
    if resolved.skipped:
        return executor.skip()
    if not resolved.success:
        error_msg = resolved.error or f"{target_name} computation failed"
        return executor.fail(RuntimeError(error_msg))

    def compute() -> RowCounts:
        return dict(resolved.table_counts)

    return executor.execute(compute)


@tag_materialize()
def executor_record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ExecutionResult,
    *,
    hash_options: InputHashOptions | None = None,
) -> TargetRunRecord:
    """Short-name wrapper for executor_materialize for subDAG wiring.

    Returns
    -------
    TargetRunRecord
        Execution record for the target.
    """
    return executor_materialize(
        env,
        graph,
        target_name,
        compute_result,
        hash_options=hash_options,
    )


def build_duckdb_materialization_module(*, target_name: str, table_key: str) -> ModuleType:
    """Build a module that materializes an Ibis expression to DuckDB.

    Parameters
    ----------
    target_name
        Target name for contract attribution and manifest hashing.
    table_key
        Output table key.

    Returns
    -------
    ModuleType
        Module exposing ``ibis_expr_to_save`` and ``duckdb_record`` nodes.
    """
    module = ModuleType(f"duckdb_materialization_{target_name}_{table_key}")

    @SaveToObjectMetadataDecorator(
        [DuckDBIbisTableSaver],
        output_name_="materialization",
        env=source("env"),
        graph=source("graph"),
        target_name=value(target_name),
        table_key=value(table_key),
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

    @tag_materialize(target=target_name)
    def duckdb_record(
        env: BuildEnv,
        graph: TargetGraph,
        materialization: MaterializationMetadata,
    ) -> TargetRunRecord:
        """Convert saver metadata into a TargetRunRecord.

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

    module.__dict__["ibis_expr_to_save"] = ibis_expr_to_save
    module.__dict__["duckdb_record"] = duckdb_record
    return module


__all__ = [
    "build_duckdb_materialization_module",
    "executor_materialize",
    "executor_record",
]
