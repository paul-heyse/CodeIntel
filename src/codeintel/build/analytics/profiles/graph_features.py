"""Graph feature helpers for analytics profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.profiles.types import FunctionGraphFeatures
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_int

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
    edges = inputs.call_graph_edges
    if edges.is_empty():
        return {}

    edges = edges.filter((pl.col("repo") == inputs.repo) & (pl.col("commit") == inputs.commit))
    if edges.is_empty():
        return {}

    cg_out = edges.group_by("caller_goid_h128").agg(
        [
            pl.len().alias("call_edge_out_count"),
            pl.col("callee_goid_h128").n_unique().alias("call_fan_out"),
        ]
    )
    cg_in = (
        edges.filter(pl.col("callee_goid_h128").is_not_null())
        .group_by("callee_goid_h128")
        .agg(
            [
                pl.len().alias("call_edge_in_count"),
                pl.col("caller_goid_h128").n_unique().alias("call_fan_in"),
            ]
        )
    )
    combined = cg_out.join(
        cg_in,
        left_on="caller_goid_h128",
        right_on="callee_goid_h128",
        how="full",
    )
    combined = combined.with_columns(
        [
            pl.coalesce("caller_goid_h128", "callee_goid_h128").alias("function_goid_h128"),
            pl.coalesce("call_fan_in", 0).alias("call_fan_in"),
            pl.coalesce("call_fan_out", 0).alias("call_fan_out"),
            pl.coalesce("call_edge_in_count", 0).alias("call_edge_in_count"),
            pl.coalesce("call_edge_out_count", 0).alias("call_edge_out_count"),
        ]
    )
    nodes = inputs.call_graph_nodes
    relation = combined.join(
        nodes,
        left_on="function_goid_h128",
        right_on="goid_h128",
        how="left",
    ).select(
        [
            "function_goid_h128",
            "call_fan_in",
            "call_fan_out",
            "call_edge_in_count",
            "call_edge_out_count",
            (pl.col("call_fan_out") == 0).alias("call_is_leaf"),
            ((pl.col("call_fan_in") == 0) & (pl.col("call_fan_out") > 0)).alias(
                "call_is_entrypoint"
            ),
            pl.col("is_public").alias("call_is_public"),
        ]
    )

    features: dict[int, FunctionGraphFeatures] = {}
    for row in relation.iter_rows(named=True):
        goid = normalize_decimal_id(row.get("function_goid_h128"))
        if goid is None:
            continue
        features[goid] = FunctionGraphFeatures(
            function_goid_h128=goid,
            call_fan_in=coerce_optional_int(row.get("call_fan_in"), ctx="call_fan_in") or 0,
            call_fan_out=coerce_optional_int(row.get("call_fan_out"), ctx="call_fan_out") or 0,
            call_edge_in_count=coerce_optional_int(
                row.get("call_edge_in_count"),
                ctx="call_edge_in_count",
            )
            or 0,
            call_edge_out_count=coerce_optional_int(
                row.get("call_edge_out_count"),
                ctx="call_edge_out_count",
            )
            or 0,
            call_is_leaf=bool(row.get("call_is_leaf")),
            call_is_entrypoint=bool(row.get("call_is_entrypoint")),
            call_is_public=bool(row.get("call_is_public")),
        )
    return features
