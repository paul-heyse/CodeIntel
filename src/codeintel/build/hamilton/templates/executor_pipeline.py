"""Reusable subDAG pipeline for NativeTargetExecutor-based targets.

This module provides a templatable materialize pattern for targets that:
1. Compute and persist data internally (returning a Result dataclass)
2. Use NativeTargetExecutor for skip/execute orchestration
3. Return TargetRunRecord with table counts

Targets using this pattern: goids, symbol_uses, call_graph, import_graph,
cfg, dfg, coverage_test_edges, behavioral_coverage, and more.

Notes
-----
Hamilton namespaces nodes created by ``@subdag`` using dotted names
(``<namespace>.<node_name>``). This is acceptable for internal pipeline nodes;
the public target node (e.g., ``t__goids``) remains a stable identifier.
"""

from __future__ import annotations

from typing import Any

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph

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
    """Materialize using NativeTargetExecutor pattern.

    This function provides the standard materialize pattern for targets that:
    1. Have a compute node returning a Result dataclass with success/error/table_counts
    2. Use NativeTargetExecutor for skip-check, timing, and manifest persistence

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.
    graph
        Target graph for looking up the target metadata.
    target_name
        Name of the target being materialized.
    compute_result
        Result from the upstream compute node implementing ComputeResult protocol.

    Returns
    -------
    TargetRunRecord
        Record with status="succeeded", "skipped", or "failed".

    Examples
    --------
    Use with ``@subdag`` to wire a compute node to this materialize pattern:

    >>> @subdag(
    ...     executor_pipeline,
    ...     inputs={
    ...         "env": source("env"),
    ...         "graph": source("graph"),
    ...         "target_name": value("goids"),
    ...         "compute_result": source("t__goids__extract"),
    ...     },
    ... )
    ... def t__goids(record: TargetRunRecord) -> TargetRunRecord:
    ...     return record
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
def record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    compute_result: ComputeResult,
) -> TargetRunRecord:
    """Alias for executor_materialize for subDAG composition.

    This function provides the same functionality as ``executor_materialize``
    but with a shorter name for use in subDAG wiring where the output node
    name matters.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.
    graph
        Target graph for looking up the target metadata.
    target_name
        Name of the target being materialized.
    compute_result
        Result from the upstream compute node implementing ComputeResult protocol.

    Returns
    -------
    TargetRunRecord
        Record with status="succeeded", "skipped", or "failed".
    """
    return executor_materialize(env, graph, target_name, compute_result)


__all__ = [
    "ComputeResult",
    "executor_materialize",
    "record",
]
