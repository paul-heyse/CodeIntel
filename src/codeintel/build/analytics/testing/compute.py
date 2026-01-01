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

from codeintel.build.analytics.compute.graphs import bipartite_degrees
from codeintel.build.analytics.testing.graph_metrics import (
    TEST_GRAPH_METRICS_FUNCTIONS_COLS,
    TEST_GRAPH_METRICS_TESTS_COLS,
    TestMetricsContext,
    _build_function_rows,
    _build_test_rows,
)
from codeintel.build.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.build.graphs.runtime.context import GraphContextSpec, resolve_graph_context
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import coerce_optional_float, iter_tuples_from_arrow_reader

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


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
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
) -> TestGraphMetricsResult:
    """Compute test graph metrics without writing to database.

    Compute graph metrics from the test-function bipartite graph, returning
    row data that can be materialized separately.

    Parameters
    ----------
    gateway
        Storage gateway for reading graph data and risk factors.
    snapshot
        Repository and commit snapshot reference.
    runtime
        Optional graph runtime or options for graph computation.

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
    resolved_options = (
        runtime.options if isinstance(runtime, GraphRuntime) else runtime
    ) or GraphRuntimeOptions()

    resolved_runtime = resolve_graph_runtime(
        gateway,
        resolved_options.snapshot or snapshot,
        resolved_options,
    )

    graph = resolved_runtime.ensure_test_function_bipartite()
    graph_ctx = resolve_graph_context(
        GraphContextSpec(
            repo=snapshot.repo,
            commit=snapshot.commit,
            use_gpu=resolved_runtime.backend.use_gpu,
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
    reader = gateway.execute(
        """
        SELECT function_goid_h128, risk_score
        FROM analytics.goid_risk_factors
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    risk_by_goid: dict[int, float] = {}
    for goid_raw, score_raw in iter_tuples_from_arrow_reader(reader):
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        risk_by_goid[goid] = coerce_optional_float(score_raw, ctx="risk_score") or 0.0
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


__all__ = [
    "TEST_GRAPH_METRICS_FUNCTIONS_COLS",
    "TEST_GRAPH_METRICS_TESTS_COLS",
    "TestGraphMetricsResult",
    "compute_test_graph_metrics_pure",
]
