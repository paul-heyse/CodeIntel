"""Pure compute functions for test graph metrics.

This module provides pure compute functions that return row data without
performing any database writes. The materialization is handled by the
Hamilton native module in `build/hamilton/native/analytics/graph_metrics_pipeline.py`.

The functions compute graph metrics from the test-function bipartite graph,
returning structured result containers that can be materialized to DuckDB
tables by the build system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import networkx as nx
import polars as pl

from codeintel.build.analytics.compute.graphs import bipartite_degrees
from codeintel.build.analytics.testing.graph_metrics import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
    TestMetricsContext,
    _build_function_rows,
    _build_test_rows,
)
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_float

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestGraphMetricsResult:
    """Result container for test graph metrics computation.

    Contains row data for both test and function metrics tables without
    performing writes. The rows are tuples matching the column specifications
    in the schema.

    Attributes
    ----------
    test_rows
        Rows for analytics.test_graph_metrics_tests table.
    function_rows
        Rows for analytics.test_graph_metrics_functions table.
    """

    test_rows: tuple[tuple[object, ...], ...]
    function_rows: tuple[tuple[object, ...], ...]


def compute_test_graph_metrics_pure(
    snapshot: SnapshotRef,
    *,
    goid_risk_factors_frame: pl.DataFrame | None = None,
) -> TestGraphMetricsResult:
    """Compute test graph metrics without writing to database.

    Compute graph metrics from the test-function bipartite graph, returning
    row data that can be materialized separately.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    goid_risk_factors_frame
        Function risk scores keyed by GOID.

    Returns
    -------
    TestGraphMetricsResult
        Container with rows for test and function metrics tables.

    Notes
    -----
    This function is a pure transformation that reads from the database but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.

    The metrics computed include:
    - Degree and weighted degree in the bipartite graph
    - Degree centrality
    - Projection metrics (clustering, betweenness)
    - Risk-weighted degree based on function risk scores
    """
    graph = nx.Graph()
    graph_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=snapshot.repo,
            commit=snapshot.commit,
            use_gpu=False,
            now=datetime.now(UTC),
            pagerank_weight="weight",
            betweenness_weight="weight",
        )
    )
    now = graph_ctx.resolved_now()

    tests = {
        cast("tuple[str, object]", node)
        for node, data in graph.nodes(data=True)
        if data.get("bipartite") == 0
    }
    funcs = {cast("tuple[str, object]", node) for node in set(graph) - tests}
    degrees = bipartite_degrees(
        graph,
        tests,
        funcs,
        weight=graph_ctx.pagerank_weight,
    )
    risk_by_goid: dict[int, float] = {}
    if goid_risk_factors_frame is not None and not goid_risk_factors_frame.is_empty():
        filtered = _filter_frame_by_snapshot(
            goid_risk_factors_frame,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        for row in filtered.iter_rows(named=True):
            goid = normalize_decimal_id(row.get("goid_h128"))
            if goid is None:
                continue
            risk_by_goid[goid] = (
                coerce_optional_float(row.get("risk_score"), ctx="risk_score") or 0.0
            )
    ctx = TestMetricsContext(
        repo=snapshot.repo,
        commit=snapshot.commit,
        now=now,
        degrees=degrees,
        risk_by_goid=risk_by_goid,
        graph_ctx=graph_ctx,
    )

    test_rows = _build_test_rows(graph, tests, ctx)
    func_rows = _build_function_rows(graph, funcs, ctx)

    log.info(
        "test graph metrics computed: %d test rows, %d function rows for %s@%s",
        len(test_rows),
        len(func_rows),
        snapshot.repo,
        snapshot.commit,
    )

    return TestGraphMetricsResult(
        test_rows=tuple(test_rows),
        function_rows=tuple(func_rows),
    )


def _filter_frame_by_snapshot(
    frame: pl.DataFrame,
    *,
    repo: str,
    commit: str,
) -> pl.DataFrame:
    filtered = frame
    if "repo" in filtered.columns:
        filtered = filtered.filter(pl.col("repo") == repo)
    if "commit" in filtered.columns:
        filtered = filtered.filter(pl.col("commit") == commit)
    return filtered


__all__ = [
    "TEST_GRAPH_METRICS_FUNCTIONS_COLS",
    "TEST_GRAPH_METRICS_TESTS_COLS",
    "TestGraphMetricsResult",
    "compute_test_graph_metrics_pure",
]
