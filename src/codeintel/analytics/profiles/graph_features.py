"""Graph feature helpers for analytics profiles."""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.analytics.profiles.types import FunctionGraphFeatures, FunctionProfileInputs


def summarize_graph_for_function_profile(
    inputs: FunctionProfileInputs,
) -> Mapping[int, FunctionGraphFeatures]:
    """
    Build graph-derived metrics keyed by function GOID.

    The computation mirrors the call-graph degree CTEs previously embedded in
    :func:`codeintel.analytics.profiles.build_function_profile`.

    Returns
    -------
    Mapping[int, FunctionGraphFeatures]
        Mapping keyed by function GOID containing call graph metrics.
    """
    con = inputs.con
    rows = con.execute(
        """
        WITH cg_out AS (
            SELECT
                caller_goid_h128 AS function_goid_h128,
                COUNT(*) AS call_edge_out_count,
                COUNT(DISTINCT callee_goid_h128) AS call_fan_out
            FROM graph.call_graph_edges
            WHERE repo = ? AND commit = ?
            GROUP BY caller_goid_h128
        ),
        cg_in AS (
            SELECT
                callee_goid_h128 AS function_goid_h128,
                COUNT(*) AS call_edge_in_count,
                COUNT(DISTINCT caller_goid_h128) AS call_fan_in
            FROM graph.call_graph_edges
            WHERE callee_goid_h128 IS NOT NULL
              AND repo = ? AND commit = ?
            GROUP BY callee_goid_h128
        ),
        cg_nodes AS (
            SELECT
                goid_h128 AS function_goid_h128,
                is_public
            FROM graph.call_graph_nodes
        )
        SELECT
            COALESCE(co.function_goid_h128, ci.function_goid_h128, cn.function_goid_h128)
                AS function_goid_h128,
            COALESCE(ci.call_fan_in, 0) AS call_fan_in,
            COALESCE(co.call_fan_out, 0) AS call_fan_out,
            COALESCE(ci.call_edge_in_count, 0) AS call_edge_in_count,
            COALESCE(co.call_edge_out_count, 0) AS call_edge_out_count,
            COALESCE(co.call_fan_out, 0) = 0 AS call_is_leaf,
            COALESCE(ci.call_fan_in, 0) = 0 AND COALESCE(co.call_fan_out, 0) > 0
                AS call_is_entrypoint,
            cn.is_public AS call_is_public
        FROM cg_out AS co
        FULL OUTER JOIN cg_in AS ci USING (function_goid_h128)
        FULL OUTER JOIN cg_nodes AS cn USING (function_goid_h128)
        """,
        [inputs.repo, inputs.commit, inputs.repo, inputs.commit],
    ).fetchall()

    features: dict[int, FunctionGraphFeatures] = {}
    for (
        function_goid_h128,
        call_fan_in,
        call_fan_out,
        call_edge_in_count,
        call_edge_out_count,
        call_is_leaf,
        call_is_entrypoint,
        call_is_public,
    ) in rows:
        goid = int(function_goid_h128)
        features[goid] = FunctionGraphFeatures(
            function_goid_h128=goid,
            call_fan_in=int(call_fan_in or 0),
            call_fan_out=int(call_fan_out or 0),
            call_edge_in_count=int(call_edge_in_count or 0),
            call_edge_out_count=int(call_edge_out_count or 0),
            call_is_leaf=bool(call_is_leaf),
            call_is_entrypoint=bool(call_is_entrypoint),
            call_is_public=bool(call_is_public),
        )
    return features
