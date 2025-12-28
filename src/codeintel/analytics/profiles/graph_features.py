"""Graph feature helpers for analytics profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.analytics.profiles.types import FunctionGraphFeatures
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.helpers.sql_params import render_sql

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.analytics.profiles.types import FunctionProfileInputs


def summarize_graph_for_function_profile(
    inputs: FunctionProfileInputs,
) -> Mapping[int, FunctionGraphFeatures]:
    """
    Build graph-derived metrics keyed by function GOID.

    The computation mirrors the call-graph degree CTEs used by
    :func:`codeintel.analytics.profiles.functions.build_function_profile_rows`.

    Returns
    -------
    Mapping[int, FunctionGraphFeatures]
        Mapping keyed by function GOID containing call graph metrics.
    """
    try:
        relation = inputs.gateway.con.sql(
            render_sql(
                """
                WITH scoped_edges AS (
                    SELECT *
                    FROM graph.call_graph_edges
                    WHERE repo = $repo AND commit = $commit
                ),
                cg_out AS (
                    SELECT
                        caller_goid_h128 AS function_goid_h128,
                        COUNT(*) AS call_edge_out_count,
                        COUNT(DISTINCT callee_goid_h128) AS call_fan_out
                    FROM scoped_edges
                    GROUP BY caller_goid_h128
                ),
                cg_in AS (
                    SELECT
                        callee_goid_h128 AS function_goid_h128,
                        COUNT(*) AS call_edge_in_count,
                        COUNT(DISTINCT caller_goid_h128) AS call_fan_in
                    FROM scoped_edges
                    WHERE callee_goid_h128 IS NOT NULL
                    GROUP BY callee_goid_h128
                ),
                combined AS (
                    SELECT
                        COALESCE(cg_out.function_goid_h128, cg_in.function_goid_h128)
                            AS function_goid_h128,
                        COALESCE(cg_in.call_fan_in, 0) AS call_fan_in,
                        COALESCE(cg_out.call_fan_out, 0) AS call_fan_out,
                        COALESCE(cg_in.call_edge_in_count, 0) AS call_edge_in_count,
                        COALESCE(cg_out.call_edge_out_count, 0) AS call_edge_out_count
                    FROM cg_out
                    FULL OUTER JOIN cg_in
                      ON cg_out.function_goid_h128 = cg_in.function_goid_h128
                )
                SELECT
                    combined.function_goid_h128,
                    combined.call_fan_in,
                    combined.call_fan_out,
                    combined.call_edge_in_count,
                    combined.call_edge_out_count,
                    combined.call_fan_out = 0 AS call_is_leaf,
                    combined.call_fan_in = 0 AND combined.call_fan_out > 0 AS call_is_entrypoint,
                    nodes.is_public AS call_is_public
                FROM combined
                LEFT JOIN graph.call_graph_nodes AS nodes
                  ON combined.function_goid_h128 = nodes.goid_h128
                ORDER BY combined.function_goid_h128
                """,
                {"repo": inputs.repo, "commit": inputs.commit},
            )
        )
    except DuckDBError:
        return {}

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
    ) in relation.fetchall():
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
