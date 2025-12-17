"""Native Hamilton implementations for graph_metrics and graph_validation targets.

This module consolidates the graph metrics and validation targets into a single
Hamilton-native module to reduce boilerplate and improve discoverability.

Targets
-------
- ``graph_metrics``: Computes graph-derived analytics tables.
- ``graph_validation``: Runs integrity checks on graph tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, SupportsInt, cast

from hamilton.function_modifiers import tag

from codeintel.analytics.graphs import (
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
)
from codeintel.analytics.graphs.graph_metrics import GraphMetricsDeps
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.targets import TargetGraph
from codeintel.config.primitives import GraphBackendConfig
from codeintel.graphs.runtime import (
    GraphMetricsOptions,
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import and_predicates, filter_by, ibis_bool

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

_GRAPH_METRICS_OUTPUT_TABLES = (
    "analytics.graph_metrics_functions",
    "analytics.graph_metrics_modules",
    "analytics.graph_metrics_functions_ext",
    "analytics.graph_metrics_modules_ext",
    "analytics.graph_stats",
)


@dataclass(frozen=True)
class GraphMetricsComputeResult:
    """Result from graph metrics computation.

    Attributes
    ----------
    success
        Whether computation completed successfully.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if computation failed.
    """

    success: bool
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def _count_rows(
    gateway: StorageGateway,
    table: str,
    repo: str,
    commit: str,
) -> int:
    """Count rows in a table for the given snapshot.

    Parameters
    ----------
    gateway
        Storage gateway.
    table
        Table name.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Row count.
    """
    try:
        tbl = gateway.ibis.table(table)
        filtered = tbl.filter(and_predicates(tbl.repo == repo, tbl.commit == commit))
        result_df = filtered.aggregate(row_count=tbl.repo.count()).execute()
        return int(result_df.iloc[0]["row_count"]) if not result_df.empty else 0
    except (RuntimeError, ValueError, OSError, KeyError):
        return 0


@tag(domain="graphs", target="graph_metrics", node_type="compute")
def t__graph_metrics__compute(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
) -> GraphMetricsComputeResult:
    """Compute graph metrics from call graph data.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__call_graph
        Upstream call_graph target result (for dependency).

    Returns
    -------
    GraphMetricsComputeResult
        Result containing table row counts.
    """
    if t__call_graph.status != "succeeded":
        return GraphMetricsComputeResult(
            success=False,
            error=f"Upstream call_graph target failed: {t__call_graph.error}",
        )

    try:
        gateway = env.gateway
        snapshot = env.snapshot
        repo, commit = snapshot.repo, snapshot.commit

        log.info(
            "graph_metrics: Computing metrics for repo=%s commit=%s",
            repo,
            commit,
        )

        backend_config = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
        runtime_options = GraphRuntimeOptions(snapshot=snapshot, backend=backend_config)
        runtime = build_graph_runtime(gateway, runtime_options)

        options = GraphMetricsOptions()
        deps = GraphMetricsDeps(
            catalog_provider=None,
            runtime=runtime,
        )
        compute_graph_metrics(gateway, snapshot, options=options, deps=deps)

        compute_graph_metrics_functions_ext(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        compute_graph_metrics_modules_ext(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        compute_graph_stats(
            gateway,
            repo=repo,
            commit=commit,
            runtime=runtime,
        )

        row_counts: dict[str, int] = {}
        for table in _GRAPH_METRICS_OUTPUT_TABLES:
            row_counts[table] = _count_rows(gateway, table, repo, commit)

        log.info("graph_metrics: Computed metrics row_counts=%s", row_counts)

        return GraphMetricsComputeResult(
            success=True,
            table_counts=row_counts,
        )

    except (RuntimeError, ValueError, OSError) as exc:
        log.exception("Graph metrics computation failed")
        return GraphMetricsComputeResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="graph_metrics", node_type="materialize")
def t__graph_metrics(
    env: BuildEnv,
    graph: TargetGraph,
    t__graph_metrics__compute: GraphMetricsComputeResult,
) -> TargetRunRecord:
    """Materialize graph metrics target with validation."""
    executor = NativeTargetExecutor.for_target(env, graph, "graph_metrics")

    if executor.should_skip():
        return executor.skip()

    if not t__graph_metrics__compute.success:
        return executor.fail(
            RuntimeError(t__graph_metrics__compute.error or "Graph metrics computation failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__graph_metrics__compute.table_counts)

    return executor.execute(compute)


@dataclass(frozen=True)
class GraphValidationResult:
    """Result from graph validation.

    Attributes
    ----------
    success
        Whether validation passed (no errors).
    error_count
        Number of validation errors found.
    errors
        List of validation error messages.
    table_counts
        Row counts per output (validation errors).
    error
        Fatal error message if validation failed.
    """

    success: bool
    error_count: int = 0
    errors: list[str] = field(default_factory=list)
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def _validate_call_graph_integrity(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[str]:
    """Validate call graph edge integrity."""
    errors: list[str] = []

    try:
        edges = gateway.ibis.table("graph.call_graph_edges")
        nodes = gateway.ibis.table("graph.call_graph_nodes")

        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        caller_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.caller_goid_h128, nodes.goid_h128)]
        )
        orphan_callers_expr = caller_join.filter(ibis_bool(nodes.goid_h128.isnull())).count()
        orphan_callers = int(cast("SupportsInt", orphan_callers_expr.execute()))
        if orphan_callers > 0:
            errors.append(f"Found {orphan_callers} call graph edges with orphan caller GOIDs")

        callee_join = scoped_edges.left_join(
            nodes, predicates=[(scoped_edges.callee_goid_h128, nodes.goid_h128)]
        )
        orphan_callees_expr = callee_join.filter(
            ibis_bool(scoped_edges.callee_goid_h128.notnull()) & ibis_bool(nodes.goid_h128.isnull())
        ).count()
        orphan_callees = int(cast("SupportsInt", orphan_callees_expr.execute()))
        if orphan_callees > 0:
            log.debug(
                "validation: %d call graph edges have unresolved callee GOIDs",
                orphan_callees,
            )
    except DuckDBError as exc:
        log.debug("validation: Could not validate call graph: %s", exc)

    return errors


@tag(node_type="helper")
def _validate_import_graph_integrity(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> list[str]:
    """Validate import graph integrity."""
    errors: list[str] = []

    try:
        edges = gateway.ibis.table("graph.import_graph_edges")
        modules = gateway.ibis.table("graph.import_modules")
        scoped_edges = filter_by(edges, edges.repo == repo, edges.commit == commit)

        joined = scoped_edges.left_join(
            modules,
            predicates=[
                (scoped_edges.src_module, modules.module),
                (scoped_edges.repo, modules.repo),
                (scoped_edges.commit, modules.commit),
            ],
        )
        orphan_src_expr = joined.filter(ibis_bool(modules.module.isnull())).count()
        orphan_src = int(cast("SupportsInt", orphan_src_expr.execute()))
        if orphan_src > 0:
            errors.append(f"Found {orphan_src} import edges with missing source modules")

    except DuckDBError as exc:
        log.debug("validation: Could not validate import graph: %s", exc)

    return errors


@tag(node_type="helper")
def _validate_cfg_integrity(
    gateway: StorageGateway,
    _repo: str,
    _commit: str,
) -> list[str]:
    """Validate CFG integrity."""
    errors: list[str] = []

    try:
        edges = gateway.ibis.table("graph.cfg_edges")
        blocks = gateway.ibis.table("graph.cfg_blocks")

        joined = edges.left_join(
            blocks,
            predicates=[
                (edges.src_block_id, blocks.block_id),
                (edges.function_goid_h128, blocks.function_goid_h128),
            ],
        )
        orphan_edges_expr = joined.filter(ibis_bool(blocks.block_id.isnull())).count()
        orphan_edges = int(cast("SupportsInt", orphan_edges_expr.execute()))
        if orphan_edges > 0:
            errors.append(f"Found {orphan_edges} CFG edges with missing source blocks")

    except DuckDBError as exc:
        log.debug("validation: Could not validate CFG: %s", exc)

    return errors


@tag(domain="graphs", target="graph_validation", node_type="compute")
def t__graph_validation__check(
    env: BuildEnv,
    t__call_graph: TargetRunRecord,
    t__import_graph: TargetRunRecord,
    t__cfg: TargetRunRecord,
) -> GraphValidationResult:
    """Run validation checks on all graph data."""
    deps = [("call_graph", t__call_graph), ("import_graph", t__import_graph), ("cfg", t__cfg)]
    for name, record in deps:
        if record.status != "succeeded":
            return GraphValidationResult(
                success=False,
                error=f"Upstream {name} target failed: {record.error}",
            )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit

        all_errors: list[str] = []

        call_graph_errors = _validate_call_graph_integrity(gateway, repo, commit)
        all_errors.extend(call_graph_errors)

        import_graph_errors = _validate_import_graph_integrity(gateway, repo, commit)
        all_errors.extend(import_graph_errors)

        cfg_errors = _validate_cfg_integrity(gateway, repo, commit)
        all_errors.extend(cfg_errors)

        for error in all_errors:
            log.warning("graph_validation: %s", error)

        log.info(
            "graph_validation: Completed with %d issues found for repo=%s commit=%s",
            len(all_errors),
            repo,
            commit,
        )

        return GraphValidationResult(
            success=len(all_errors) == 0,
            error_count=len(all_errors),
            errors=all_errors,
            table_counts={"analytics.graph_validation": len(all_errors)},
        )

    except Exception as exc:
        log.exception("Graph validation failed")
        return GraphValidationResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="graph_validation", node_type="materialize")
def t__graph_validation(
    env: BuildEnv,
    graph: TargetGraph,
    t__graph_validation__check: GraphValidationResult,
) -> TargetRunRecord:
    """Materialize graph validation target."""
    executor = NativeTargetExecutor.for_target(env, graph, "graph_validation")

    if executor.should_skip():
        return executor.skip()

    if t__graph_validation__check.error:
        return executor.fail(RuntimeError(t__graph_validation__check.error))

    if not t__graph_validation__check.success:
        errors_msg = "\n".join(t__graph_validation__check.errors)
        return executor.fail(RuntimeError(f"Graph validation failed:\n{errors_msg}"))

    def compute() -> dict[str, int]:
        return dict(t__graph_validation__check.table_counts)

    return executor.execute(compute)


__all__ = [
    "GraphMetricsComputeResult",
    "GraphValidationResult",
    "t__graph_metrics",
    "t__graph_metrics__compute",
    "t__graph_validation",
    "t__graph_validation__check",
]

