"""Graph feature helpers for analytics profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.analytics.profiles.types import FunctionGraphFeatures
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.storage.duckdb_types import ColumnExpression, ConstantExpression
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.query_results import coerce_optional_int, iter_tuples_from_relation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.analytics.profiles.types import FunctionProfileInputs


def summarize_graph_for_function_profile(
    inputs: FunctionProfileInputs,
) -> Mapping[int, FunctionGraphFeatures]:
    """
    Build graph-derived metrics keyed by function GOID.

    The computation mirrors the call-graph degree CTEs used by
    :func:`codeintel.build.analytics.profiles.functions.build_function_profile_rows`.

    Returns
    -------
    Mapping[int, FunctionGraphFeatures]
        Mapping keyed by function GOID containing call graph metrics.
    """
    try:
        predicate = (ColumnExpression("repo") == ConstantExpression(inputs.repo)) & (
            ColumnExpression("commit") == ConstantExpression(inputs.commit)
        )
        edges = (
            inputs.gateway.relation_from_table_key("graph.call_graph_edges")
            .filter(predicate)
            .set_alias("edges")
        )
        cg_out = edges.aggregate(
            "count(*) as call_edge_out_count, count(distinct callee_goid_h128) as call_fan_out",
            "caller_goid_h128",
        ).set_alias("cg_out")
        cg_in = (
            edges.filter(~ColumnExpression("callee_goid_h128").isnull())
            .aggregate(
                "count(*) as call_edge_in_count, count(distinct caller_goid_h128) as call_fan_in",
                "callee_goid_h128",
            )
            .set_alias("cg_in")
        )
        combined = cg_out.join(
            cg_in,
            "cg_out.caller_goid_h128 = cg_in.callee_goid_h128",
            how="full",
        ).set_alias("combined")
        combined = combined.select(
            "coalesce(cg_out.caller_goid_h128, cg_in.callee_goid_h128) as function_goid_h128",
            "coalesce(cg_in.call_fan_in, 0) as call_fan_in",
            "coalesce(cg_out.call_fan_out, 0) as call_fan_out",
            "coalesce(cg_in.call_edge_in_count, 0) as call_edge_in_count",
            "coalesce(cg_out.call_edge_out_count, 0) as call_edge_out_count",
        ).set_alias("combined")
        nodes = inputs.gateway.relation_from_table_key("graph.call_graph_nodes").set_alias("nodes")
        relation = (
            combined.join(
                nodes,
                "combined.function_goid_h128 = nodes.goid_h128",
                how="left",
            )
            .select(
                "combined.function_goid_h128",
                "combined.call_fan_in",
                "combined.call_fan_out",
                "combined.call_edge_in_count",
                "combined.call_edge_out_count",
                "combined.call_fan_out = 0 as call_is_leaf",
                "combined.call_fan_in = 0 AND combined.call_fan_out > 0 as call_is_entrypoint",
                "nodes.is_public as call_is_public",
            )
            .order("combined.function_goid_h128")
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
    ) in iter_tuples_from_relation(relation):
        goid = normalize_decimal_id(function_goid_h128)
        if goid is None:
            continue
        features[goid] = FunctionGraphFeatures(
            function_goid_h128=goid,
            call_fan_in=coerce_optional_int(call_fan_in, ctx="call_fan_in") or 0,
            call_fan_out=coerce_optional_int(call_fan_out, ctx="call_fan_out") or 0,
            call_edge_in_count=coerce_optional_int(
                call_edge_in_count,
                ctx="call_edge_in_count",
            )
            or 0,
            call_edge_out_count=coerce_optional_int(
                call_edge_out_count,
                ctx="call_edge_out_count",
            )
            or 0,
            call_is_leaf=bool(call_is_leaf),
            call_is_entrypoint=bool(call_is_entrypoint),
            call_is_public=bool(call_is_public),
        )
    return features
