"""Graph feature helpers for analytics profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ibis

from codeintel.analytics.profiles.types import FunctionGraphFeatures
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.ibis_types import filter_by

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.analytics.profiles.types import FunctionProfileInputs


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
    try:
        ibis_api = cast("Any", inputs.gateway.ibis)
        edges = ibis_api.table("graph.call_graph_edges")
        nodes = ibis_api.table("graph.call_graph_nodes")
    except DuckDBError:
        return {}

    scoped_edges = filter_by(edges, edges.repo == inputs.repo, edges.commit == inputs.commit)

    cg_out = (
        scoped_edges.group_by(scoped_edges.caller_goid_h128)
        .aggregate(
            call_edge_out_count=scoped_edges.caller_goid_h128.count(),
            call_fan_out=scoped_edges.callee_goid_h128.nunique(),
        )
        .rename({"caller_goid_h128": "function_goid_h128"})
    )

    cg_in = (
        scoped_edges.filter(scoped_edges.callee_goid_h128.notnull())
        .group_by(scoped_edges.callee_goid_h128)
        .aggregate(
            call_edge_in_count=scoped_edges.callee_goid_h128.count(),
            call_fan_in=scoped_edges.caller_goid_h128.nunique(),
        )
        .rename({"callee_goid_h128": "function_goid_h128"})
    )

    cg_nodes = nodes.select(
        nodes.goid_h128.name("function_goid_h128"),
        nodes.is_public,
    )

    joined = cg_out.outer_join(
        cg_in, [cg_out.function_goid_h128 == cg_in.function_goid_h128]
    ).outer_join(cg_nodes, [cg_out.function_goid_h128 == cg_nodes.function_goid_h128])

    rows_df = (
        joined.select(
            joined.function_goid_h128,
            joined.call_fan_in.fillna(ibis.literal(0)),
            joined.call_fan_out.fillna(ibis.literal(0)),
            joined.call_edge_in_count.fillna(ibis.literal(0)),
            joined.call_edge_out_count.fillna(ibis.literal(0)),
            (joined.call_fan_out.fillna(ibis.literal(0)) == 0).name("call_is_leaf"),
            (
                (joined.call_fan_in.fillna(ibis.literal(0)) == 0)
                & (joined.call_fan_out.fillna(ibis.literal(0)) > 0)
            ).name("call_is_entrypoint"),
            joined.is_public.name("call_is_public"),
        )
        .order_by(joined.function_goid_h128)
        .execute()
    )

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
    ) in rows_df.itertuples(index=False, name=None):
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
