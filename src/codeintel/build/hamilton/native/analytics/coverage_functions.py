"""Native Hamilton implementation for coverage_functions target.

This module implements the coverage_functions analytics target as a pure Hamilton DAG,
computing per-function coverage metrics by joining GOIDs with coverage line data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import duckdb
import ibis
from hamilton.function_modifiers import tag

from codeintel.build.hamilton.manifest_hook import compute_target_input_hash
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_table
from codeintel.build.hamilton.native.runner import (
    NativeRunInfo,
    create_failed_record,
    create_skipped_record,
    create_success_record,
    save_manifest,
    should_skip_native_target,
)
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.build.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph


@tag(domain="analytics", target="coverage_functions", node_kind="compute")
def t__coverage_functions__compute(
    env: BuildEnv,
    q__graph__goids: ir.Table,
    q__analytics__coverage_lines: ir.Table,
) -> ir.Table:
    """Compute per-function coverage metrics from GOIDs and coverage lines.

    This node joins function GOIDs with coverage line data to aggregate:
    - Total executable lines per function
    - Covered lines per function
    - Coverage ratio (covered/executable)

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__graph__goids
        Ibis table expression for graph.goids (function definitions).
    q__analytics__coverage_lines
        Ibis table expression for analytics.coverage_lines (line-level coverage).

    Returns
    -------
    ir.Table
        Ibis expression for analytics.coverage_functions with schema:
        - function_goid_h128: Function GOID hash
        - urn: Function URN
        - repo: Repository
        - commit: Commit hash
        - rel_path: Relative file path
        - language: Programming language
        - kind: Function kind
        - qualname: Qualified name
        - start_line: Function start line
        - end_line: Function end line
        - executable_lines: Count of executable lines
        - covered_lines: Count of covered lines
        - coverage_ratio: Coverage percentage (0.0-1.0)

    Examples
    --------
    >>> # This node is executed by Hamilton as part of the coverage_functions target
    >>> # It produces an Ibis expression that is materialized by t__coverage_functions
    """
    LOG.info("Computing coverage_functions: joining GOIDs with coverage lines")

    # Filter GOIDs to the current snapshot
    goids_filtered = q__graph__goids.filter(
        cast(
            "Any",
            and_predicates(
                q__graph__goids.repo == env.snapshot.repo,
                q__graph__goids.commit == env.snapshot.commit,
            ),
        )
    )

    # Filter coverage lines to the current snapshot
    coverage_filtered = q__analytics__coverage_lines.filter(
        cast(
            "Any",
            and_predicates(
                q__analytics__coverage_lines.repo == env.snapshot.repo,
                q__analytics__coverage_lines.commit == env.snapshot.commit,
            ),
        )
    )

    # Aggregate coverage lines per function
    # Group by function_goid_h128 and sum executable/covered lines
    coverage_agg = coverage_filtered.group_by("function_goid_h128").aggregate(
        executable_lines=cast("Any", coverage_filtered.executable_lines).sum(),
        covered_lines=cast("Any", coverage_filtered.covered_lines).sum(),
    )

    # Join GOIDs with aggregated coverage
    result = goids_filtered.left_join(
        coverage_agg, predicates=[goids_filtered.goid_h128 == coverage_agg.function_goid_h128]
    )

    # Compute coverage ratio (handle null/zero cases)
    result = result.mutate(
        coverage_ratio=(
            cast("Any", result.covered_lines).cast("float64")
            / cast("Any", result.executable_lines).cast("float64")
        ).fillna(ibis.literal(0.0))
    )

    # Select and rename columns to match analytics.coverage_functions schema
    coverage_functions = result.select(
        function_goid_h128=result.goid_h128,
        urn=result.urn,
        repo=result.repo,
        commit=result.commit,
        rel_path=result.rel_path,
        language=result.language,
        kind=result.kind,
        qualname=result.qualname,
        start_line=result.start_line,
        end_line=result.end_line,
        executable_lines=cast("Any", result.executable_lines).fillna(ibis.literal(0)),
        covered_lines=cast("Any", result.covered_lines).fillna(ibis.literal(0)),
        coverage_ratio=result.coverage_ratio,
    )

    LOG.info("coverage_functions compute complete")
    return coverage_functions


@tag(domain="analytics", target="coverage_functions", node_kind="materialize")
def t__coverage_functions(
    env: BuildEnv,
    graph: TargetGraph,
    t__coverage_functions__compute: ir.Table,
) -> TargetRunRecord:
    """Materialize coverage_functions compute result to DuckDB.

    This node takes the Ibis expression from the compute node and writes it to
    analytics.coverage_functions, creating a DatasetRef for lineage tracking.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    t__coverage_functions__compute
        Ibis expression for coverage_functions from compute node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.

    Examples
    --------
    >>> # This node is executed by Hamilton after the compute node succeeds
    >>> # It materializes the Ibis expression to DuckDB and returns a TargetRunRecord
    """
    LOG.info("Materializing coverage_functions to DuckDB")

    target = graph.get("coverage_functions")
    if target is None:
        return create_failed_record(
            target=graph.get("modules") or graph.all_targets[0],
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            error=ValueError("coverage_functions target not found in graph"),
        )

    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        manifests=env.manifest_index,
    )

    if should_skip_native_target(env, target, input_hash):
        return create_skipped_record(
            target=target,
            env=env,
            run=NativeRunInfo(input_hash=input_hash, options_hash=None, duration_ms=0.0),
        )

    try:
        ref = materialize_table(
            MaterializationContext(
                gateway=env.gateway,
                snapshot=env.snapshot,
                validate=env.validate_outputs,
                owner_target=target.name,
                input_hash=input_hash,
            ),
            "analytics.coverage_functions",
            t__coverage_functions__compute,
        )
    except (ValueError, RuntimeError, duckdb.Error) as exc:
        return create_failed_record(
            target=target,
            input_hash=input_hash,
            options_hash=None,
            duration_ms=0.0,
            error=exc,
        )

    record = create_success_record(
        target=target,
        env=env,
        run=NativeRunInfo(
            input_hash=input_hash,
            options_hash=None,
            duration_ms=0.0,
            row_counts={"analytics.coverage_functions": ref.row_count or 0},
        ),
    )
    save_manifest(env, record)
    return record


__all__ = ["t__coverage_functions", "t__coverage_functions__compute"]
