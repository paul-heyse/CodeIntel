"""Native Hamilton implementation for hotspots target.

This module implements the hotspots analytics target as a pure Hamilton DAG,
computing file hotspot metrics based on code churn and complexity analysis.
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


@tag(domain="analytics", target="hotspots", node_kind="compute")
def t__hotspots__compute(
    env: BuildEnv,
    q__core__modules: ir.Table,
    q__core__file_state: ir.Table,
) -> ir.Table:
    """Compute file hotspot metrics from module and file state data.

    This node analyzes code churn patterns and file complexity to identify
    hotspots in the codebase. Hotspots are files with high change frequency
    and complexity, indicating areas that may need refactoring attention.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    q__core__modules
        Ibis table expression for core.modules (file/module metadata).
    q__core__file_state
        Ibis table expression for core.file_state (git churn data).

    Returns
    -------
    ir.Table
        Ibis expression for analytics.hotspots with schema:
        - rel_path: Relative file path
        - commit_count: Number of commits touching this file
        - author_count: Number of unique authors
        - lines_added: Total lines added across commits
        - lines_deleted: Total lines deleted across commits
        - complexity: File complexity metric (e.g., cyclomatic complexity sum)
        - score: Hotspot score (weighted combination of metrics)

    Examples
    --------
    >>> # This node is executed by Hamilton as part of the hotspots target
    >>> # It produces an Ibis expression that is materialized by t__hotspots
    """
    LOG.info("Computing hotspots: analyzing file churn and complexity")

    # Filter modules to the current snapshot
    modules_filtered = q__core__modules.filter(
        cast(
            "Any",
            and_predicates(
                q__core__modules.repo == env.snapshot.repo,
                q__core__modules.commit == env.snapshot.commit,
            ),
        )
    )

    # Filter file_state to the current snapshot
    file_state_filtered = q__core__file_state.filter(
        cast(
            "Any",
            and_predicates(
                q__core__file_state.repo == env.snapshot.repo,
                q__core__file_state.commit == env.snapshot.commit,
            ),
        )
    )

    # Aggregate churn metrics per file from file_state
    churn_metrics = file_state_filtered.group_by("rel_path").aggregate(
        commit_count=ibis._.count(),
        lines_added=cast("Any", file_state_filtered.lines_added).sum().fillna(ibis.literal(0)),
        lines_deleted=cast("Any", file_state_filtered.lines_deleted).sum().fillna(ibis.literal(0)),
        # Count distinct authors if author column exists
        author_count=(
            cast("Any", file_state_filtered.author).nunique()
            if "author" in cast("Any", file_state_filtered).columns
            else ibis.literal(1)
        ),
    )

    # Extract complexity from modules (if available)
    # For now, use a placeholder complexity metric based on module size
    modules_complexity = modules_filtered.group_by("rel_path").aggregate(
        complexity=cast("Any", cast("Any", modules_filtered.loc).sum())
        .fillna(ibis.literal(0))
        .cast("float64"),
    )

    # Join churn metrics with complexity
    hotspots = churn_metrics.left_join(
        modules_complexity, predicates=[churn_metrics.rel_path == modules_complexity.rel_path]
    )

    # Compute hotspot score as a weighted combination of churn and complexity signals.
    lines_sum = cast("Any", hotspots.lines_added) + cast("Any", hotspots.lines_deleted)
    hotspots = hotspots.mutate(
        score=(
            (cast("Any", hotspots.commit_count).cast("float64") * 0.4)
            + (cast("Any", hotspots.author_count).cast("float64") * 0.2)
            + (cast("Any", lines_sum).cast("float64") / ibis.literal(1000.0) * 0.2)
            + (cast("Any", hotspots.complexity) / ibis.literal(100.0) * 0.2)
        ).fillna(ibis.literal(0.0))
    )

    # Select columns to match analytics.hotspots schema
    result = hotspots.select(
        rel_path=churn_metrics.rel_path,
        commit_count=hotspots.commit_count,
        author_count=hotspots.author_count,
        lines_added=hotspots.lines_added,
        lines_deleted=hotspots.lines_deleted,
        complexity=cast("Any", hotspots.complexity).fillna(ibis.literal(0.0)),
        score=hotspots.score,
    )

    LOG.info("hotspots compute complete")
    return result


@tag(domain="analytics", target="hotspots", node_kind="materialize")
def t__hotspots(
    env: BuildEnv,
    graph: TargetGraph,
    t__hotspots__compute: ir.Table,
) -> TargetRunRecord:
    """Materialize hotspots compute result to DuckDB.

    This node takes the Ibis expression from the compute node and writes it to
    analytics.hotspots, creating a DatasetRef for lineage tracking.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    t__hotspots__compute
        Ibis expression for hotspots from compute node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.

    Examples
    --------
    >>> # This node is executed by Hamilton after the compute node succeeds
    >>> # It materializes the Ibis expression to DuckDB and returns a TargetRunRecord
    """
    LOG.info("Materializing hotspots to DuckDB")

    target = graph.get("hotspots")
    if target is None:
        return create_failed_record(
            target=graph.get("modules") or graph.all_targets[0],
            input_hash="",
            options_hash=None,
            duration_ms=0.0,
            error=ValueError("hotspots target not found in graph"),
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
            "analytics.hotspots",
            t__hotspots__compute,
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
            row_counts={"analytics.hotspots": ref.row_count or 0},
        ),
    )
    save_manifest(env, record)
    return record


__all__ = ["t__hotspots", "t__hotspots__compute"]
