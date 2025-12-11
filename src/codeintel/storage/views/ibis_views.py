"""Ibis-defined views for analytics and core datasets.

This module provides view builders registered via the Ibis view registry.
Each builder function takes an IbisGateway and returns an Ibis table expression.

The legacy create_* functions are maintained for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ibis
import ibis.expr.types as it

from codeintel.storage.ibis_types import ne, or_predicates
from codeintel.storage.views.ibis_registry import register_view

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.ibis_adapter import IbisGateway


def _table(con: DuckDBBackend, qualified_name: str) -> it.Table:
    """Return table using database qualifier when provided.

    Parameters
    ----------
    con
        Ibis DuckDB backend connection.
    qualified_name
        Table name, optionally schema-qualified (e.g., "analytics.function_metrics").

    Returns
    -------
    it.Table
        Ibis table expression.
    """
    if "." in qualified_name:
        database, table = qualified_name.split(".", 1)
        return con.table(table, database=database)
    return con.table(qualified_name)


def _create_view(con: DuckDBBackend, qualified_name: str, expr: it.Table) -> None:
    """Create view using database qualifier when provided.

    Ibis 11+ requires the `database` parameter for schema-qualified names.
    This method automatically splits "schema.view" into the correct form.

    Parameters
    ----------
    con
        Ibis DuckDB backend connection.
    qualified_name
        View name, optionally schema-qualified (e.g., "analytics.v_function_summary").
    expr
        Ibis table expression defining the view.
    """
    if "." in qualified_name:
        database, view_name = qualified_name.split(".", 1)
        con.create_view(view_name, expr, database=database, overwrite=True)
    else:
        con.create_view(qualified_name, expr, overwrite=True)


__all__ = [
    "build_call_graph_enriched",
    "build_callgraph_degree",
    "build_docs_file_summary",
    "build_docs_function_summary",
    "build_docs_module_architecture",
    "build_docs_subsystem_coverage",
    "build_docs_subsystem_profile",
    "build_docs_subsystem_summary",
    "build_function_hotspots",
    "build_function_summary",
    "build_goid_crosswalk_join",
    "build_goid_crosswalk_mismatches",
    "build_import_graph_degree",
    "create_all_ibis_views",
    "create_call_graph_enriched_view",
    "create_callgraph_degree_view",
    "create_docs_file_summary_view",
    "create_docs_function_summary_view",
    "create_docs_module_architecture_view",
    "create_docs_subsystem_coverage_view",
    "create_docs_subsystem_profile_view",
    "create_docs_subsystem_summary_view",
    "create_function_hotspots_view",
    "create_function_summary_view",
    "create_goid_crosswalk_views",
    "create_import_graph_degree_view",
]

CALLGRAPH_LOC_SMALL = 50
CALLGRAPH_LOC_MEDIUM = 200
COMPLEXITY_LOW_MAX = 5
COMPLEXITY_MEDIUM_MAX = 10


# ---------------------------------------------------------------------------
# View Builders (registered via decorator)
# ---------------------------------------------------------------------------


@register_view("analytics.v_function_summary")
def build_function_summary(ibis_gw: IbisGateway) -> it.Table:
    """Build the function summary view expression.

    Combines metrics with typedness details and adds lightweight derived
    buckets for complexity and LOC.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    fm: it.Table = _table(con, "analytics.function_metrics")
    ft: it.Table = _table(con, "analytics.function_types").select(
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

    loc_col = cast("it.NumericValue", joined["loc"])
    cyclomatic_complexity = cast("it.NumericValue", joined.cyclomatic_complexity)
    small_loc = loc_col <= ibis.literal(CALLGRAPH_LOC_SMALL)
    medium_loc = loc_col <= ibis.literal(CALLGRAPH_LOC_MEDIUM)
    low_complexity = cyclomatic_complexity <= ibis.literal(COMPLEXITY_LOW_MAX)
    medium_complexity = cyclomatic_complexity <= ibis.literal(COMPLEXITY_MEDIUM_MAX)
    loc_bucket = ibis.cases(
        (small_loc, "small"),
        (medium_loc, "medium"),
        else_="large",
    )
    complexity_band = ibis.cases(
        (low_complexity, "low"),
        (medium_complexity, "medium"),
        else_="high",
    )

    enriched = joined.mutate(
        loc_bucket=loc_bucket,
        complexity_band=complexity_band,
    )
    return enriched.select(
        enriched.function_goid_h128,
        enriched.repo,
        enriched.commit,
        enriched.rel_path,
        enriched.language,
        enriched.kind,
        enriched.qualname,
        enriched["loc"],
        enriched.logical_loc,
        enriched.param_count,
        enriched.positional_params,
        enriched.keyword_only_params,
        enriched.has_varargs,
        enriched.has_varkw,
        enriched.is_async,
        enriched.is_generator,
        enriched.return_count,
        enriched.yield_count,
        enriched.raise_count,
        enriched.cyclomatic_complexity,
        enriched.complexity_bucket,
        enriched.complexity_band,
        enriched.max_nesting_depth,
        enriched.stmt_count,
        enriched.decorator_count,
        enriched.has_docstring,
        enriched.created_at,
        enriched.loc_bucket,
        enriched.param_typed_ratio,
        enriched.typedness_bucket,
        enriched.typedness_source,
        enriched.return_type,
        enriched.has_return_annotation,
    )


@register_view("docs.v_function_summary")
def build_docs_function_summary(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_function_summary from risk factors and metrics.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    rf: it.Table = _table(con, "analytics.goid_risk_factors")
    fm: it.Table = _table(con, "analytics.function_metrics")

    joined = rf.left_join(
        fm,
        [
            rf.function_goid_h128 == fm.function_goid_h128,
            rf.repo == fm.repo,
            rf.commit == fm.commit,
        ],
    )
    return joined.select(
        rf.function_goid_h128,
        rf.urn,
        rf.repo,
        rf.commit,
        rf.rel_path,
        rf.language,
        rf.kind,
        rf.qualname,
        rf.loc,
        rf.logical_loc,
        rf.cyclomatic_complexity,
        rf.complexity_bucket,
        fm.param_count,
        fm.positional_params,
        fm.keyword_only_params,
        fm.has_varargs,
        fm.has_varkw,
        fm.is_async,
        fm.is_generator,
        fm.return_count,
        fm.yield_count,
        fm.raise_count,
        rf.typedness_bucket,
        rf.typedness_source,
        rf.hotspot_score,
        rf.file_typed_ratio,
        rf.static_error_count,
        rf.has_static_errors,
        rf.executable_lines,
        rf.covered_lines,
        rf.coverage_ratio,
        rf.tested,
        rf.test_count,
        rf.failing_test_count,
        rf.last_test_status,
        rf.risk_score,
        rf.risk_level,
        rf.tags,
        rf.owners,
        rf.created_at,
    )


@register_view("graph.v_call_graph_degree")
def build_callgraph_degree(ibis_gw: IbisGateway) -> it.Table:
    """Build the call graph degree view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    edges: it.Table = _table(con, "graph.call_graph_edges")

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

    repo_value = ibis.coalesce(out_degree.repo, in_degree.repo)
    commit_value = ibis.coalesce(out_degree.commit, in_degree.commit)
    function_goid = ibis.coalesce(out_degree.caller_goid_h128, in_degree.callee_goid_h128)
    call_out_degree_val = ibis.coalesce(out_degree.call_out_degree, ibis.literal(0))
    call_in_degree_val = ibis.coalesce(in_degree.call_in_degree, ibis.literal(0))

    return joined.select(
        repo=repo_value,
        commit=commit_value,
        function_goid_h128=function_goid,
        call_out_degree=call_out_degree_val,
        call_in_degree=call_in_degree_val,
    )


@register_view("core.v_goid_crosswalk_join")
def build_goid_crosswalk_join(ibis_gw: IbisGateway) -> it.Table:
    """Build the goid crosswalk join view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    goids: it.Table = _table(con, "core.goids")
    crosswalk: it.Table = _table(con, "core.goid_crosswalk")

    joined = goids.inner_join(
        crosswalk,
        [
            goids.repo == crosswalk.repo,
            goids.commit == crosswalk.commit,
            goids.urn == crosswalk.goid,
        ],
    )

    return joined.select(
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


@register_view("core.v_goid_crosswalk_mismatches")
def build_goid_crosswalk_mismatches(ibis_gw: IbisGateway) -> it.Table:
    """Build the goid crosswalk mismatches view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    goids: it.Table = _table(con, "core.goids")
    crosswalk: it.Table = _table(con, "core.goid_crosswalk")

    joined = goids.inner_join(
        crosswalk,
        [
            goids.repo == crosswalk.repo,
            goids.commit == crosswalk.commit,
            goids.urn == crosswalk.goid,
        ],
    )

    return joined.filter(
        or_predicates(
            ne(goids.language, crosswalk.lang),
            ne(goids.rel_path, crosswalk.file_path),
            ne(goids.qualname, crosswalk.ast_qualname),
        )
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


@register_view("analytics.v_function_hotspots")
def build_function_hotspots(ibis_gw: IbisGateway) -> it.Table:
    """Build the function hotspots view with normalized scores.

    Normalizes hotspot_score to [0,1] for relative comparisons.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    rf: it.Table = _table(con, "analytics.goid_risk_factors")
    min_score = rf.hotspot_score.min()
    max_score = rf.hotspot_score.max()
    score_range = max_score.cast("float64") - min_score.cast("float64")
    normalized_score = ibis.cases(
        (score_range == 0, 0.0),
        else_=(rf.hotspot_score.cast("float64") - min_score.cast("float64")) / score_range,
    )
    hotspots = rf.mutate(
        hotspot_normalized=normalized_score,
    )
    return hotspots.select(
        rf.function_goid_h128,
        rf.repo,
        rf.commit,
        rf.rel_path,
        rf.language,
        rf.kind,
        rf.qualname,
        rf.hotspot_score,
        normalized_score.name("hotspot_normalized"),
        rf.cyclomatic_complexity,
        rf.coverage_ratio,
        rf.risk_score,
        rf.risk_level,
        rf.complexity_bucket,
        rf.typedness_bucket,
        rf.created_at,
    )


@register_view("graph.v_import_graph_degree")
def build_import_graph_degree(ibis_gw: IbisGateway) -> it.Table:
    """Build the import graph degree view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    edges: it.Table = _table(con, "graph.import_graph_edges")

    out_degree = edges.group_by(["repo", "commit", "src_module"]).aggregate(
        import_out_degree=edges.dst_module.count()
    )
    in_degree = edges.group_by(["repo", "commit", "dst_module"]).aggregate(
        import_in_degree=edges.src_module.count()
    )

    joined = out_degree.outer_join(
        in_degree,
        [
            out_degree.repo == in_degree.repo,
            out_degree.commit == in_degree.commit,
            out_degree.src_module == in_degree.dst_module,
        ],
    )
    return joined.select(
        repo=ibis.coalesce(out_degree.repo, in_degree.repo),
        commit=ibis.coalesce(out_degree.commit, in_degree.commit),
        module=ibis.coalesce(out_degree.src_module, in_degree.dst_module),
        import_out_degree=ibis.coalesce(out_degree.import_out_degree, ibis.literal(0)),
        import_in_degree=ibis.coalesce(in_degree.import_in_degree, ibis.literal(0)),
    )


@register_view("docs.v_call_graph_enriched")
def build_call_graph_enriched(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_call_graph_enriched view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    edges: it.Table = _table(con, "graph.call_graph_edges")
    goids: it.Table = _table(con, "core.goids")
    risk: it.Table = _table(con, "analytics.goid_risk_factors")

    caller_goids = goids.view()
    callee_goids = goids.view()
    caller_risk = risk.view()
    callee_risk = risk.view()

    joined = (
        edges.left_join(
            caller_goids,
            [
                edges.caller_goid_h128 == caller_goids.goid_h128,
                edges.repo == caller_goids.repo,
                edges.commit == caller_goids.commit,
            ],
        )
        .left_join(
            callee_goids,
            [
                edges.callee_goid_h128 == callee_goids.goid_h128,
                edges.repo == callee_goids.repo,
                edges.commit == callee_goids.commit,
            ],
        )
        .left_join(
            caller_risk,
            [
                edges.caller_goid_h128 == caller_risk.function_goid_h128,
                edges.repo == caller_risk.repo,
                edges.commit == caller_risk.commit,
            ],
        )
        .left_join(
            callee_risk,
            [
                edges.callee_goid_h128 == callee_risk.function_goid_h128,
                edges.repo == callee_risk.repo,
                edges.commit == callee_risk.commit,
            ],
        )
    )

    return joined.select(
        edges.caller_goid_h128,
        caller_goids.repo.name("caller_repo"),
        caller_goids.commit.name("caller_commit"),
        caller_goids.urn.name("caller_urn"),
        caller_goids.rel_path.name("caller_rel_path"),
        caller_goids.qualname.name("caller_qualname"),
        caller_risk.risk_level.name("caller_risk_level"),
        caller_risk.risk_score.name("caller_risk_score"),
        edges.callee_goid_h128,
        callee_goids.repo.name("callee_repo"),
        callee_goids.commit.name("callee_commit"),
        callee_goids.urn.name("callee_urn"),
        callee_goids.rel_path.name("callee_rel_path"),
        callee_goids.qualname.name("callee_qualname"),
        callee_risk.risk_level.name("callee_risk_level"),
        callee_risk.risk_score.name("callee_risk_score"),
        edges.callsite_path,
        edges.callsite_line,
        edges.callsite_col,
        edges.language,
        edges.kind,
        edges.resolved_via,
        edges.confidence,
        edges.evidence_json,
    )


@register_view("docs.v_file_summary")
def build_docs_file_summary(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_file_summary view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    fp: it.Table = _table(con, "analytics.file_profile")
    modules: it.Table = _table(con, "core.modules")

    joined = fp.left_join(
        modules,
        [
            fp.repo == modules.repo,
            fp.commit == modules.commit,
            fp.rel_path == modules.path,
        ],
    )

    return joined.select(
        fp.repo,
        fp.commit,
        fp.rel_path,
        modules.module.name("module"),
        fp.language,
        fp.function_count,
        fp.class_count,
        fp.avg_loc.name("loc"),
        fp.avg_cyclomatic_complexity.name("complexity"),
        fp.max_risk_score.name("avg_risk_score"),
        fp.max_risk_score,
        fp.high_risk_function_count,
        fp.file_coverage_ratio.name("coverage_ratio"),
        fp.annotation_ratio.name("typed_ratio"),
        fp.hotspot_score,
        fp.static_error_count,
        modules.tags,
        modules.owners,
    )


@register_view("docs.v_module_architecture")
def build_docs_module_architecture(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_module_architecture view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    modules: it.Table = _table(con, "core.modules")
    graph_metrics: it.Table = _table(con, "analytics.graph_metrics_modules")
    subsystem_modules: it.Table = _table(con, "analytics.subsystem_modules")
    subsystems: it.Table = _table(con, "analytics.subsystems")

    joined = (
        modules.left_join(
            graph_metrics,
            [
                modules.repo == graph_metrics.repo,
                modules.commit == graph_metrics.commit,
                modules.module == graph_metrics.module,
            ],
        )
        .left_join(
            subsystem_modules,
            [
                modules.repo == subsystem_modules.repo,
                modules.commit == subsystem_modules.commit,
                modules.module == subsystem_modules.module,
            ],
        )
        .left_join(
            subsystems,
            [
                subsystem_modules.repo == subsystems.repo,
                subsystem_modules.commit == subsystems.commit,
                subsystem_modules.subsystem_id == subsystems.subsystem_id,
            ],
        )
    )

    return joined.select(
        modules.repo,
        modules.commit,
        modules.module,
        modules.path.name("rel_path"),
        graph_metrics.import_fan_in,
        graph_metrics.import_fan_out,
        graph_metrics.import_pagerank,
        graph_metrics.symbol_fan_in,
        graph_metrics.symbol_fan_out,
        subsystem_modules.subsystem_id,
        subsystems.name.name("subsystem_name"),
        modules.tags,
        modules.owners,
    )


@register_view("docs.v_subsystem_summary")
def build_docs_subsystem_summary(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_subsystem_summary view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    subsystems: it.Table = _table(con, "analytics.subsystems")
    profile: it.Table = _table(con, "analytics.subsystem_profile_cache")

    joined = subsystems.left_join(
        profile,
        [
            subsystems.repo == profile.repo,
            subsystems.commit == profile.commit,
            subsystems.subsystem_id == profile.subsystem_id,
        ],
    )

    return joined.select(
        subsystems.repo,
        subsystems.commit,
        subsystems.subsystem_id,
        subsystems.name,
        subsystems.description,
        subsystems.module_count,
        subsystems.modules_json,
        subsystems.entrypoints_json,
        subsystems.internal_edge_count,
        subsystems.external_edge_count,
        subsystems.fan_in,
        subsystems.fan_out,
        profile.function_count,
        profile.avg_risk_score,
        profile.max_risk_score,
        profile.high_risk_function_count,
        profile.risk_level,
        subsystems.created_at,
    )


@register_view("docs.v_subsystem_profile")
def build_docs_subsystem_profile(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_subsystem_profile view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    subsystems: it.Table = _table(con, "analytics.subsystems")
    profile: it.Table = _table(con, "analytics.subsystem_profile_cache")
    graph_metrics: it.Table = _table(con, "analytics.subsystem_graph_metrics")

    joined = subsystems.left_join(
        profile,
        [
            subsystems.repo == profile.repo,
            subsystems.commit == profile.commit,
            subsystems.subsystem_id == profile.subsystem_id,
        ],
    ).left_join(
        graph_metrics,
        [
            subsystems.repo == graph_metrics.repo,
            subsystems.commit == graph_metrics.commit,
            subsystems.subsystem_id == graph_metrics.subsystem_id,
        ],
    )

    return joined.select(
        subsystems.repo,
        subsystems.commit,
        subsystems.subsystem_id,
        subsystems.name,
        subsystems.description,
        subsystems.module_count,
        subsystems.modules_json,
        subsystems.entrypoints_json,
        subsystems.internal_edge_count,
        subsystems.external_edge_count,
        subsystems.fan_in,
        subsystems.fan_out,
        profile.function_count,
        profile.avg_risk_score,
        profile.max_risk_score,
        profile.high_risk_function_count,
        profile.risk_level,
        graph_metrics.import_in_degree,
        graph_metrics.import_out_degree,
        graph_metrics.import_pagerank,
        graph_metrics.import_betweenness,
        graph_metrics.import_closeness,
        graph_metrics.import_layer,
        subsystems.created_at,
    )


@register_view("docs.v_subsystem_coverage")
def build_docs_subsystem_coverage(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_subsystem_coverage view.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    con = ibis_gw.con
    subsystems: it.Table = _table(con, "analytics.subsystems")
    profile: it.Table = _table(con, "analytics.subsystem_profile_cache")
    coverage: it.Table = _table(con, "analytics.subsystem_coverage_cache")

    joined = subsystems.left_join(
        profile,
        [
            subsystems.repo == profile.repo,
            subsystems.commit == profile.commit,
            subsystems.subsystem_id == profile.subsystem_id,
        ],
    ).left_join(
        coverage,
        [
            subsystems.repo == coverage.repo,
            subsystems.commit == coverage.commit,
            subsystems.subsystem_id == coverage.subsystem_id,
        ],
    )

    return joined.select(
        subsystems.repo,
        subsystems.commit,
        subsystems.subsystem_id,
        subsystems.name,
        subsystems.description,
        subsystems.module_count,
        profile.function_count,
        profile.risk_level,
        profile.avg_risk_score,
        profile.max_risk_score,
        coverage.test_count,
        coverage.passed_test_count,
        coverage.failed_test_count,
        coverage.skipped_test_count,
        coverage.xfail_test_count,
        coverage.flaky_test_count,
        coverage.total_functions_covered,
        coverage.avg_functions_covered,
        coverage.max_functions_covered,
        coverage.min_functions_covered,
        coverage.function_coverage_ratio,
        subsystems.created_at,
    )


# ---------------------------------------------------------------------------
# Legacy create_* functions (for backward compatibility)
# ---------------------------------------------------------------------------


def create_function_summary_view(gateway: StorageGateway) -> None:
    """Create or replace analytics.v_function_summary using Ibis expressions."""
    expr = build_function_summary(gateway.ibis)
    _create_view(gateway.ibis.con, "analytics.v_function_summary", expr)


def create_docs_function_summary_view(gateway: StorageGateway) -> None:
    """Create docs.v_function_summary derived from risk factors and metrics."""
    expr = build_docs_function_summary(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_function_summary", expr)


def create_callgraph_degree_view(gateway: StorageGateway) -> None:
    """Create or replace graph.v_call_graph_degree using call graph edges."""
    expr = build_callgraph_degree(gateway.ibis)
    _create_view(gateway.ibis.con, "graph.v_call_graph_degree", expr)


def create_goid_crosswalk_views(gateway: StorageGateway) -> None:
    """Create or replace goid crosswalk QA views."""
    join_expr = build_goid_crosswalk_join(gateway.ibis)
    _create_view(gateway.ibis.con, "core.v_goid_crosswalk_join", join_expr)

    mismatch_expr = build_goid_crosswalk_mismatches(gateway.ibis)
    _create_view(gateway.ibis.con, "core.v_goid_crosswalk_mismatches", mismatch_expr)


def create_function_hotspots_view(gateway: StorageGateway) -> None:
    """Create analytics.v_function_hotspots using goid risk factors."""
    expr = build_function_hotspots(gateway.ibis)
    _create_view(gateway.ibis.con, "analytics.v_function_hotspots", expr)


def create_import_graph_degree_view(gateway: StorageGateway) -> None:
    """Create graph.v_import_graph_degree aggregating import edge degrees."""
    expr = build_import_graph_degree(gateway.ibis)
    _create_view(gateway.ibis.con, "graph.v_import_graph_degree", expr)


def create_call_graph_enriched_view(gateway: StorageGateway) -> None:
    """Create docs.v_call_graph_enriched to align with the SQL definition."""
    expr = build_call_graph_enriched(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_call_graph_enriched", expr)


def create_docs_file_summary_view(gateway: StorageGateway) -> None:
    """Create docs.v_file_summary aggregating per-file statistics."""
    expr = build_docs_file_summary(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_file_summary", expr)


def create_docs_module_architecture_view(gateway: StorageGateway) -> None:
    """Create docs.v_module_architecture combining module metrics with subsystem."""
    expr = build_docs_module_architecture(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_module_architecture", expr)


def create_docs_subsystem_summary_view(gateway: StorageGateway) -> None:
    """Create docs.v_subsystem_summary combining subsystem with profile cache."""
    expr = build_docs_subsystem_summary(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_subsystem_summary", expr)


def create_docs_subsystem_profile_view(gateway: StorageGateway) -> None:
    """Create docs.v_subsystem_profile with full profile and graph metrics."""
    expr = build_docs_subsystem_profile(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_subsystem_profile", expr)


def create_docs_subsystem_coverage_view(gateway: StorageGateway) -> None:
    """Create docs.v_subsystem_coverage combining subsystem with coverage data."""
    expr = build_docs_subsystem_coverage(gateway.ibis)
    _create_view(gateway.ibis.con, "docs.v_subsystem_coverage", expr)


def create_all_ibis_views(gateway: StorageGateway) -> None:
    """Create Ibis-defined views that supplement (not replace) SQL views.

    This function creates ONLY views that:
    1. Do not exist in SQL form (unique to Ibis)
    2. Require runtime computation (e.g., min/max normalization)

    Views that already have complete SQL definitions in views/*.py are NOT
    recreated here to avoid schema mismatches between SQL and Ibis versions.
    The SQL views in subsystem_views.py, function_views.py, module_views.py,
    and graph_views.py are the source of truth for docs.* views.
    """
    # Unique Ibis views (no SQL equivalent)
    create_function_summary_view(gateway)  # analytics.v_function_summary
    create_callgraph_degree_view(gateway)  # analytics.v_callgraph_degree
    create_goid_crosswalk_views(gateway)  # core.v_goid_crosswalk_*
    create_function_hotspots_view(gateway)  # analytics.v_function_hotspots
    create_import_graph_degree_view(gateway)  # analytics.v_import_degree

    # NOTE: The following view creators are intentionally REMOVED because
    # they overwrite complete SQL views with incomplete Ibis versions:
    # - create_docs_function_summary_view (use function_views.py instead)
    # - create_call_graph_enriched_view (use graph_views.py instead)
    # - create_docs_file_summary_view (use module_views.py instead)
    # - create_docs_module_architecture_view (use module_views.py instead)
    # - create_docs_subsystem_summary_view (use subsystem_views.py instead)
    # - create_docs_subsystem_profile_view (use subsystem_views.py instead)
    # - create_docs_subsystem_coverage_view (use subsystem_views.py instead)
