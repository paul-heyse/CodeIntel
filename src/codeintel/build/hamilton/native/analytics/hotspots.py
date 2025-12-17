"""Native Hamilton implementation for hotspots target.

This module implements the hotspots analytics target as a pure Hamilton DAG,
computing file hotspot metrics based on code churn and complexity analysis.

Includes Hamilton-native validation via @check_output_custom (Phase 1.5 POC)
and schema documentation via @schema.output.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import ibis
import ibis.expr.types as ir
from hamilton.function_modifiers import (
    check_output_custom,
    pipe_input,
    schema,
    source,
    step,
    tag,
    value,
)
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materialization,
)
from codeintel.build.hamilton.validators import build_table_contract
from codeintel.build.targets import TargetGraph
from codeintel.storage.ibis_types import and_predicates

LOG = logging.getLogger(__name__)
_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, ir.Table)


@tag(domain="analytics", target="hotspots", node_type="compute")
def hotspots__modules_complexity(env: BuildEnv, q__core__modules: ir.Table) -> ir.Table:
    """Compute per-file complexity proxy from core.modules.

    Parameters
    ----------
    env
        Build environment providing the snapshot for filtering.
    q__core__modules
        Ibis table expression for core.modules.

    Returns
    -------
    ir.Table
        Ibis expression with columns:
        - rel_path
        - complexity
    """
    modules_filtered = q__core__modules.filter(
        cast(
            "Any",
            and_predicates(
                q__core__modules.repo == env.snapshot.repo,
                q__core__modules.commit == env.snapshot.commit,
            ),
        )
    )
    return modules_filtered.group_by("rel_path").aggregate(
        complexity=cast("Any", cast("Any", modules_filtered.loc).sum())
        .fillna(ibis.literal(0))
        .cast("float64"),
    )


def _hotspots_filter_file_state(file_state: ir.Table, env: BuildEnv) -> ir.Table:
    """Filter core.file_state to the current snapshot.

    Parameters
    ----------
    file_state
        Ibis table expression for core.file_state.
    env
        Build environment providing the snapshot for filtering.

    Returns
    -------
    ir.Table
        Snapshot-filtered file_state expression.
    """
    return file_state.filter(
        cast(
            "Any",
            and_predicates(
                file_state.repo == env.snapshot.repo,
                file_state.commit == env.snapshot.commit,
            ),
        )
    )


def _hotspots_aggregate_churn(file_state: ir.Table) -> ir.Table:
    """Aggregate churn metrics per file from core.file_state.

    Parameters
    ----------
    file_state
        Snapshot-filtered file_state expression.

    Returns
    -------
    ir.Table
        Ibis expression with churn metrics keyed by rel_path.
    """
    return file_state.group_by("rel_path").aggregate(
        commit_count=ibis._.count(),
        lines_added=cast("Any", file_state.lines_added).sum().fillna(ibis.literal(0)),
        lines_deleted=cast("Any", file_state.lines_deleted).sum().fillna(ibis.literal(0)),
        author_count=(
            cast("Any", file_state.author).nunique()
            if "author" in cast("Any", file_state).columns
            else ibis.literal(1)
        ),
    )


def _hotspots_join_complexity(churn_metrics: ir.Table, modules_complexity: ir.Table) -> ir.Table:
    """Join churn metrics with per-file complexity.

    Parameters
    ----------
    churn_metrics
        Per-file churn metrics keyed by rel_path.
    modules_complexity
        Per-file complexity proxy keyed by rel_path.

    Returns
    -------
    ir.Table
        Joined expression combining churn and complexity columns.
    """
    return churn_metrics.left_join(
        modules_complexity,
        predicates=[churn_metrics.rel_path == modules_complexity.rel_path],
    )


def _hotspots_score(hotspots: ir.Table) -> ir.Table:
    """Compute the hotspot score as a weighted combination of signals.

    Parameters
    ----------
    hotspots
        Joined churn+complexity table keyed by rel_path.

    Returns
    -------
    ir.Table
        Expression with a computed ``score`` column.
    """
    lines_sum = cast("Any", hotspots.lines_added) + cast("Any", hotspots.lines_deleted)
    return hotspots.mutate(
        score=(
            (cast("Any", hotspots.commit_count).cast("float64") * 0.4)
            + (cast("Any", hotspots.author_count).cast("float64") * 0.2)
            + (cast("Any", lines_sum).cast("float64") / ibis.literal(1000.0) * 0.2)
            + (cast("Any", hotspots.complexity) / ibis.literal(100.0) * 0.2)
        ).fillna(ibis.literal(0.0))
    )


def _hotspots_select(hotspots: ir.Table) -> ir.Table:
    """Select final columns to match analytics.hotspots schema.

    Parameters
    ----------
    hotspots
        Table expression with churn metrics, complexity, and score.

    Returns
    -------
    ir.Table
        Final expression matching the analytics.hotspots contract schema.
    """
    return hotspots.select(
        rel_path=hotspots.rel_path,
        commit_count=hotspots.commit_count,
        author_count=hotspots.author_count,
        lines_added=hotspots.lines_added,
        lines_deleted=hotspots.lines_deleted,
        complexity=cast("Any", hotspots.complexity).fillna(ibis.literal(0.0)),
        score=hotspots.score,
    )


@SaveToDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node("analytics.hotspots"),
    env=source("env"),
    graph=source("graph"),
    target_name=value("hotspots"),
    table_key=value("analytics.hotspots"),
)
@pipe_input(
    step(_hotspots_filter_file_state, env=source("env")),
    step(_hotspots_aggregate_churn),
    step(_hotspots_join_complexity, modules_complexity=source("hotspots__modules_complexity")),
    step(_hotspots_score),
    step(_hotspots_select),
    namespace=None,
    on_input="q__core__file_state",
)
@check_output_custom(
    *build_table_contract(
        required_columns=[
            "rel_path",
            "commit_count",
            "author_count",
            "lines_added",
            "lines_deleted",
            "complexity",
            "score",
        ],
        no_nulls=["rel_path"],
    )
)
@tag(domain="analytics", target="hotspots", node_type="compute", target_="t__hotspots__compute")
@schema.output(
    ("rel_path", "string"),
    ("commit_count", "int"),
    ("author_count", "int"),
    ("lines_added", "int"),
    ("lines_deleted", "int"),
    ("complexity", "float"),
    ("score", "float"),
    target_="t__hotspots__compute",
)
def t__hotspots__compute(
    q__core__file_state: ir.Table,
) -> ir.Table:
    """Compute file hotspot metrics from module and file state data.

    This node analyzes code churn patterns and file complexity to identify
    hotspots in the codebase. Hotspots are files with high change frequency
    and complexity, indicating areas that may need refactoring attention.

    Parameters
    ----------
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
    LOG.info("Computing hotspots with DAG-visible pipe steps")
    return q__core__file_state


@tag(domain="analytics", target="hotspots", node_type="materialize")
def t__hotspots(
    env: BuildEnv,
    graph: TargetGraph,
    m__analytics__hotspots: dict[str, Any],
) -> TargetRunRecord:
    """Finalize hotspots execution from DAG-visible DuckDB materialization.

    The DuckDB write is performed by a Hamilton materializer node
    (``m__analytics__hotspots``). This target node converts the materialization
    metadata into a TargetRunRecord and persists the manifest on success.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and config.
    graph
        Target graph for accessing OutputTarget contract.
    m__analytics__hotspots
        Materialization metadata dict produced by the DuckDB saver node.

    Returns
    -------
    TargetRunRecord
        Record capturing execution status, duration, and output references.

    Examples
    --------
    >>> # This node runs after the materializer node and emits a TargetRunRecord.
    """
    return record_from_duckdb_materialization(
        env=env,
        graph=graph,
        target_name="hotspots",
        expected_table_key="analytics.hotspots",
        materialization=m__analytics__hotspots,
    )


__all__ = ["t__hotspots", "t__hotspots__compute"]
