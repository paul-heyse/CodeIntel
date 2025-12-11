"""Ibis-defined views for analytics and core datasets."""

from __future__ import annotations

import ibis
import ibis.expr.types as it

from codeintel.storage.gateway.protocol import StorageGateway

__all__ = [
    "create_all_ibis_views",
    "create_callgraph_degree_view",
    "create_function_summary_view",
    "create_goid_crosswalk_views",
]


def create_function_summary_view(gateway: StorageGateway) -> None:
    """
    Create or replace analytics.v_function_summary using Ibis expressions.

    Combines metrics with typedness details and adds lightweight derived
    buckets for complexity and LOC.
    """
    con = gateway.ibis.con
    fm: it.Table = con.table("analytics.function_metrics")
    ft: it.Table = con.table("analytics.function_types").select(
        "function_goid_h128",
        "repo",
        "commit",
        "rel_path",
        "typedness_bucket",
        "typedness_source",
        "return_type",
        "has_return_annotation",
        "param_typed_ratio",
    )

    joined = fm.left_join(
        ft,
        ["repo", "commit", "rel_path", "function_goid_h128"],
    )

    loc_bucket = (
        ibis.case()
        .when(joined.loc <= 50, "small")
        .when(joined.loc <= 200, "medium")
        .else_("large")
        .end()
    )
    complexity_band = (
        ibis.case()
        .when(joined.cyclomatic_complexity <= 5, "low")
        .when(joined.cyclomatic_complexity <= 10, "medium")
        .else_("high")
        .end()
    )

    summary = joined.mutate(
        loc_bucket=loc_bucket,
        complexity_band=complexity_band,
    ).select(
        joined.function_goid_h128,
        joined.repo,
        joined.commit,
        joined.rel_path,
        joined.language,
        joined.kind,
        joined.qualname,
        joined.loc,
        joined.logical_loc,
        joined.param_count,
        joined.positional_params,
        joined.keyword_only_params,
        joined.has_varargs,
        joined.has_varkw,
        joined.is_async,
        joined.is_generator,
        joined.return_count,
        joined.yield_count,
        joined.raise_count,
        joined.cyclomatic_complexity,
        joined.complexity_bucket,
        joined.complexity_band,
        joined.max_nesting_depth,
        joined.stmt_count,
        joined.decorator_count,
        joined.has_docstring,
        joined.created_at,
        joined.loc_bucket,
        joined.param_typed_ratio,
        joined.typedness_bucket,
        joined.typedness_source,
        joined.return_type,
        joined.has_return_annotation,
    )

    con.create_view("analytics.v_function_summary", summary, overwrite=True)


def create_callgraph_degree_view(gateway: StorageGateway) -> None:
    """Create or replace graph.v_call_graph_degree using call graph edges."""
    con = gateway.ibis.con
    edges: it.Table = con.table("graph.call_graph_edges")

    out_degree = edges.group_by(["repo", "commit", "caller_goid_h128"]).aggregate(
        call_out_degree=edges.callee_goid_h128.count()
    )
    in_degree = edges.group_by(["repo", "commit", "callee_goid_h128"]).aggregate(
        call_in_degree=edges.caller_goid_h128.count()
    )

    joined = out_degree.outer_join(
        in_degree,
        [
            out_degree.repo == in_degree.repo,
            out_degree.commit == in_degree.commit,
            out_degree.caller_goid_h128 == in_degree.callee_goid_h128,
        ],
    )

    degree_view = joined.select(
        repo=out_degree.repo.fillna(in_degree.repo),
        commit=out_degree.commit.fillna(in_degree.commit),
        function_goid_h128=out_degree.caller_goid_h128.fillna(in_degree.callee_goid_h128),
        call_out_degree=out_degree.call_out_degree.fillna(0),
        call_in_degree=in_degree.call_in_degree.fillna(0),
    )

    con.create_view("graph.v_call_graph_degree", degree_view, overwrite=True)


def create_goid_crosswalk_views(gateway: StorageGateway) -> None:
    """Create or replace goid crosswalk QA views."""
    con = gateway.ibis.con
    goids: it.Table = con.table("core.goids")
    crosswalk: it.Table = con.table("core.goid_crosswalk")

    joined = goids.inner_join(
        crosswalk,
        [
            goids.repo == crosswalk.repo,
            goids.commit == crosswalk.commit,
            goids.urn == crosswalk.goid,
        ],
    )

    joined_view = joined.select(
        goids.repo.name("repo"),
        goids.commit.name("commit"),
        goids.goid_h128,
        goids.urn,
        goids.rel_path,
        goids.language,
        goids.kind,
        goids.qualname,
        goids.start_line,
        goids.end_line,
        crosswalk.lang.name("crosswalk_lang"),
        crosswalk.module_path,
        crosswalk.file_path,
        crosswalk.ast_qualname,
        crosswalk.scip_symbol,
        crosswalk.updated_at,
    )
    con.create_view("core.v_goid_crosswalk_join", joined_view, overwrite=True)

    mismatches = joined.filter(
        (goids.language != crosswalk.lang)
        | (goids.rel_path != crosswalk.file_path)
        | (goids.qualname != crosswalk.ast_qualname)
    ).select(
        goids.repo.name("repo"),
        goids.commit.name("commit"),
        goids.goid_h128,
        goids.urn,
        crosswalk.goid.name("crosswalk_urn"),
        goids.language.name("goid_language"),
        crosswalk.lang.name("crosswalk_language"),
        goids.rel_path.name("goid_rel_path"),
        crosswalk.file_path.name("crosswalk_file_path"),
        goids.qualname.name("goid_qualname"),
        crosswalk.ast_qualname.name("crosswalk_qualname"),
        crosswalk.updated_at,
    )
    con.create_view("core.v_goid_crosswalk_mismatches", mismatches, overwrite=True)


def create_all_ibis_views(gateway: StorageGateway) -> None:
    """Create all Ibis-defined views backed by the storage gateway."""
    create_function_summary_view(gateway)
    create_callgraph_degree_view(gateway)
    create_goid_crosswalk_views(gateway)
