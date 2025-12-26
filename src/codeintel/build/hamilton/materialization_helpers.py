"""Materialization helpers for native Hamilton targets."""

from __future__ import annotations

import ibis.expr.types as ir

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_materialize
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph

# Keep types available for Hamilton's runtime type resolution.
_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    ExecutionResult,
    TargetGraph,
    TargetRunRecord,
    ir.Table,
)


@tag_materialize(domain="core")
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

    if compute_result.skipped:
        return executor.skip()
    if not compute_result.success:
        error_msg = compute_result.error or f"{target_name} computation failed"
        return executor.fail(RuntimeError(error_msg))

    def compute() -> dict[str, int]:
        return dict(compute_result.table_counts)

    return executor.execute(compute)


__all__ = ["executor_materialize"]
