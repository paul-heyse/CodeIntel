"""Ibis-defined views for analytics and core datasets.

This module provides view builders registered via the Ibis view registry.
Each builder function takes an IbisGateway and returns an Ibis table expression.

The legacy create_* functions are maintained for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ibis

from codeintel.storage.ibis_types import ne, or_predicates
from codeintel.storage.views.ibis_registry import register_view

if TYPE_CHECKING:
    import ibis.expr.types as it
    from ibis.backends.duckdb import Backend as DuckDBBackend

    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.ibis_adapter import IbisGateway

# Re-export Any for use in cast() within view builders
_ = Any  # Ensure Any is used to avoid F401


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
    # View builders
    "build_call_graph_enriched",
    "build_callgraph_degree",
    "build_docs_behavioral_classification_input",
    "build_docs_cfg_block_architecture",
    "build_docs_config_data_flow",
    "build_docs_data_model_fields",
    "build_docs_data_model_relationships",
    "build_docs_data_model_usage",
    "build_docs_data_models",
    "build_docs_data_models_normalized",
    "build_docs_dfg_block_architecture",
    "build_docs_entrypoints",
    "build_docs_external_dependencies",
    "build_docs_external_dependency_calls",
    "build_docs_file_summary",
    "build_docs_function_architecture",
    "build_docs_function_history",
    "build_docs_function_history_timeseries",
    "build_docs_function_summary",
    "build_docs_ide_hints",
    "build_docs_module_architecture",
    "build_docs_module_architecture_full",
    "build_docs_module_history_timeseries",
    "build_docs_module_with_subsystem",
    "build_docs_subsystem_agreement",
    "build_docs_subsystem_coverage",
    "build_docs_subsystem_profile",
    "build_docs_subsystem_summary",
    "build_docs_symbol_module_graph",
    "build_docs_test_architecture",
    "build_docs_test_to_function",
    "build_docs_validation_summary",
    "build_function_hotspots",
    "build_function_summary",
    "build_goid_crosswalk_join",
    "build_goid_crosswalk_mismatches",
    "build_import_graph_degree",
    # Legacy creators (deprecated)
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
    complexity_bucket = ibis.coalesce(rf.complexity_bucket, ibis.literal("unknown")).name(
        "complexity_bucket"
    )
    typedness_bucket = ibis.coalesce(rf.typedness_bucket, ibis.literal("unknown")).name(
        "typedness_bucket"
    )
    typedness_source = ibis.coalesce(rf.typedness_source, ibis.literal("unknown")).name(
        "typedness_source"
    )
    last_test_status = ibis.coalesce(rf.last_test_status, ibis.literal("untested")).name(
        "last_test_status"
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
        complexity_bucket,
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
        typedness_bucket,
        typedness_source,
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
        last_test_status,
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
    """Build docs.v_file_summary aggregating per-file statistics.

    Uses inline subquery for per-file risk aggregation to avoid CTE name conflicts.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    # Use raw SQL with inline subquery instead of CTE to avoid naming conflicts
    sql_expr = """
        SELECT
            m.repo,
            m.commit,
            m.path          AS rel_path,
            m.module,
            m.language,
            m.tags,
            m.owners,
            am.node_count,
            am.function_count,
            am.class_count,
            am.avg_depth,
            am.max_depth,
            am.complexity   AS ast_complexity,
            h.score         AS hotspot_score,
            ty.type_error_count,
            ty.annotation_ratio,
            ty.untyped_defs,
            ty.overlay_needed,
            sd.total_errors,
            sd.has_errors,
            r.function_count AS risk_function_count,
            r.high_risk_functions,
            r.medium_risk_functions,
            r.low_risk_functions,
            r.max_risk_score
        FROM core.modules m
        LEFT JOIN core.ast_metrics am
          ON am.rel_path = m.path
        LEFT JOIN analytics.hotspots h
          ON h.rel_path = m.path
        LEFT JOIN analytics.typedness ty
          ON ty.path = m.path
        LEFT JOIN analytics.static_diagnostics sd
          ON sd.rel_path = m.path
        LEFT JOIN (
            SELECT
                repo,
                commit,
                rel_path,
                COUNT(*) AS function_count,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_functions,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_risk_functions,
                SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END) AS low_risk_functions,
                MAX(risk_score) AS max_risk_score
            FROM analytics.goid_risk_factors
            GROUP BY repo, commit, rel_path
        ) AS r
          ON r.rel_path = m.path
         AND r.repo = m.repo
         AND r.commit = m.commit
    """
    return ibis_gw.con.sql(sql_expr)


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
    """Build docs.v_subsystem_summary with agreement stats.

    Includes agreement stats computed from subsystem_modules and subsystem_agreement
    using an inline subquery to avoid CTE naming conflicts.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    # Use raw SQL with inline subquery for the complex aggregation
    sql_expr = """
        SELECT
            s.repo,
            s.commit,
            s.subsystem_id,
            s.name,
            s.description,
            s.module_count,
            s.modules_json,
            coalesce(s.entrypoints_json, '[]') AS entrypoints_json,
            s.internal_edge_count,
            s.external_edge_count,
            s.fan_in,
            s.fan_out,
            s.function_count,
            s.avg_risk_score,
            s.max_risk_score,
            s.high_risk_function_count,
            s.risk_level,
            coalesce(agree.disagree_count, 0) AS subsystem_disagree_count,
            coalesce(agree.total_members, 0) AS subsystem_member_count,
            CASE
                WHEN coalesce(agree.total_members, 0) = 0 THEN NULL
                ELSE 1.0 - (coalesce(agree.disagree_count, 0) * 1.0 / agree.total_members)
            END AS subsystem_agreement_ratio,
            s.created_at
        FROM analytics.subsystems s
        LEFT JOIN (
            SELECT
                sm.repo,
                sm.commit,
                sm.subsystem_id,
                COUNT(*) AS total_members,
                SUM(CASE WHEN sa.agrees = false THEN 1 ELSE 0 END) AS disagree_count
            FROM analytics.subsystem_modules sm
            LEFT JOIN analytics.subsystem_agreement sa
              ON sa.module = sm.module
             AND sa.repo = sm.repo
             AND sa.commit = sm.commit
            GROUP BY sm.repo, sm.commit, sm.subsystem_id
        ) AS agree
          ON agree.repo = s.repo
         AND agree.commit = s.commit
         AND agree.subsystem_id = s.subsystem_id
    """
    return ibis_gw.con.sql(sql_expr)


@register_view("docs.v_subsystem_profile")
def build_docs_subsystem_profile(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_subsystem_profile with cache preference via coalesce.

    This view prefers cached values over base subsystem values using coalesce.
    Uses raw SQL for the complex coalesce/GREATEST logic.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    sql_expr = """
        SELECT
            s.repo,
            s.commit,
            s.subsystem_id,
            coalesce(c.name, s.name) AS name,
            coalesce(c.description, s.description) AS description,
            GREATEST(coalesce(c.module_count, s.module_count, 0), 0) AS module_count,
            coalesce(c.modules_json, s.modules_json) AS modules_json,
            coalesce(c.entrypoints_json, s.entrypoints_json, '[]') AS entrypoints_json,
            GREATEST(coalesce(c.internal_edge_count, s.internal_edge_count, 0), 0) AS internal_edge_count,
            GREATEST(coalesce(c.external_edge_count, s.external_edge_count, 0), 0) AS external_edge_count,
            GREATEST(coalesce(c.fan_in, s.fan_in, 0), 0) AS fan_in,
            GREATEST(coalesce(c.fan_out, s.fan_out, 0), 0) AS fan_out,
            GREATEST(coalesce(c.function_count, s.function_count, 0), 0) AS function_count,
            coalesce(c.avg_risk_score, s.avg_risk_score) AS avg_risk_score,
            coalesce(c.max_risk_score, s.max_risk_score) AS max_risk_score,
            GREATEST(coalesce(
                c.high_risk_function_count,
                s.high_risk_function_count,
                0
            ), 0) AS high_risk_function_count,
            coalesce(c.risk_level, s.risk_level) AS risk_level,
            coalesce(c.import_in_degree, gm.import_in_degree) AS import_in_degree,
            coalesce(c.import_out_degree, gm.import_out_degree) AS import_out_degree,
            coalesce(c.import_pagerank, gm.import_pagerank) AS import_pagerank,
            coalesce(c.import_betweenness, gm.import_betweenness) AS import_betweenness,
            coalesce(c.import_closeness, gm.import_closeness) AS import_closeness,
            GREATEST(coalesce(c.import_layer, gm.import_layer, 0), 0) AS import_layer,
            coalesce(c.created_at, s.created_at) AS created_at
        FROM analytics.subsystems s
        LEFT JOIN analytics.subsystem_profile_cache c
          ON c.repo = s.repo
         AND c.commit = s.commit
         AND c.subsystem_id = s.subsystem_id
        LEFT JOIN analytics.subsystem_graph_metrics gm
          ON gm.repo = s.repo
         AND gm.commit = s.commit
         AND gm.subsystem_id = s.subsystem_id
        WHERE s.repo IS NOT NULL
          AND s.commit IS NOT NULL
    """
    return ibis_gw.con.sql(sql_expr)


@register_view("docs.v_subsystem_coverage")
def build_docs_subsystem_coverage(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_subsystem_coverage with CTE and cache preference.

    This view uses a CTE to compute coverage stats from test_profile,
    then joins with subsystems and coverage cache, preferring cached values.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    # Use inline subquery instead of CTE to avoid name conflicts
    sql_expr = """
        SELECT
            s.repo,
            s.commit,
            s.subsystem_id,
            coalesce(cc.name, s.name) AS name,
            coalesce(cc.description, s.description) AS description,
            GREATEST(coalesce(cc.module_count, s.module_count, 0), 0) AS module_count,
            GREATEST(coalesce(cc.function_count, s.function_count, 0), 0) AS function_count,
            coalesce(cc.risk_level, s.risk_level) AS risk_level,
            coalesce(cc.avg_risk_score, s.avg_risk_score) AS avg_risk_score,
            coalesce(cc.max_risk_score, s.max_risk_score) AS max_risk_score,
            GREATEST(coalesce(cc.test_count, cov.test_count, 0), 0) AS test_count,
            GREATEST(coalesce(cc.passed_test_count, cov.passed_test_count, 0), 0) AS passed_test_count,
            GREATEST(coalesce(cc.failed_test_count, cov.failed_test_count, 0), 0) AS failed_test_count,
            GREATEST(coalesce(cc.skipped_test_count, cov.skipped_test_count, 0), 0) AS skipped_test_count,
            GREATEST(coalesce(cc.xfail_test_count, cov.xfail_test_count, 0), 0) AS xfail_test_count,
            GREATEST(coalesce(cc.flaky_test_count, cov.flaky_test_count, 0), 0) AS flaky_test_count,
            GREATEST(coalesce(
                cc.total_functions_covered,
                cov.total_functions_covered,
                0
            ), 0) AS total_functions_covered,
            GREATEST(coalesce(cc.avg_functions_covered, cov.avg_functions_covered, 0), 0) AS avg_functions_covered,
            GREATEST(coalesce(cc.max_functions_covered, cov.max_functions_covered, 0), 0) AS max_functions_covered,
            GREATEST(coalesce(cc.min_functions_covered, cov.min_functions_covered, 0), 0) AS min_functions_covered,
            CASE
                WHEN GREATEST(coalesce(cc.function_count, s.function_count, 0), 0) = 0 THEN NULL
                ELSE GREATEST(coalesce(cc.total_functions_covered, cov.total_functions_covered, 0), 0) * 1.0
                     / GREATEST(coalesce(cc.function_count, s.function_count, 0), 0)
            END AS function_coverage_ratio,
            coalesce(cc.created_at, s.created_at) AS created_at
        FROM analytics.subsystems s
        LEFT JOIN (
            SELECT
                repo,
                commit,
                primary_subsystem_id AS subsystem_id,
                COUNT(*) AS test_count,
                SUM(CASE WHEN status = 'passed' THEN 1 ELSE 0 END) AS passed_test_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_test_count,
                SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped_test_count,
                SUM(CASE WHEN status = 'xfail' THEN 1 ELSE 0 END) AS xfail_test_count,
                SUM(CASE WHEN coalesce(flaky, FALSE) THEN 1 ELSE 0 END) AS flaky_test_count,
                SUM(coalesce(functions_covered_count, 0)) AS total_functions_covered,
                AVG(coalesce(functions_covered_count, 0)) AS avg_functions_covered,
                MAX(coalesce(functions_covered_count, 0)) AS max_functions_covered,
                MIN(coalesce(functions_covered_count, 0)) AS min_functions_covered
            FROM analytics.test_profile
            WHERE primary_subsystem_id IS NOT NULL
            GROUP BY repo, commit, primary_subsystem_id
        ) AS cov
          ON cov.repo = s.repo
         AND cov.commit = s.commit
         AND cov.subsystem_id = s.subsystem_id
        LEFT JOIN analytics.subsystem_coverage_cache cc
          ON cc.repo = s.repo
         AND cc.commit = s.commit
         AND cc.subsystem_id = s.subsystem_id
        WHERE s.repo IS NOT NULL
          AND s.commit IS NOT NULL
    """
    return ibis_gw.con.sql(sql_expr)


@register_view("docs.v_function_architecture")
def build_docs_function_architecture(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_function_architecture with 9+ table joins.

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
    fp: it.Table = _table(con, "analytics.function_profile").view()
    gm: it.Table = _table(con, "analytics.graph_metrics_functions").view()
    gmx: it.Table = _table(con, "analytics.graph_metrics_functions_ext").view()
    tgf: it.Table = _table(con, "analytics.test_graph_metrics_functions").view()
    cfg_fn: it.Table = _table(con, "analytics.cfg_function_metrics").view()
    dfg_fn: it.Table = _table(con, "analytics.dfg_function_metrics").view()
    mp: it.Table = _table(con, "analytics.module_profile").view()
    sm: it.Table = _table(con, "analytics.subsystem_modules").view()
    ss: it.Table = _table(con, "analytics.subsystems").view()

    joined = (
        fp.left_join(
            gm,
            [
                gm.function_goid_h128 == fp.function_goid_h128,
                gm.repo == fp.repo,
                gm.commit == fp.commit,
            ],
        )
        .left_join(
            gmx,
            [
                gmx.function_goid_h128 == fp.function_goid_h128,
                gmx.repo == fp.repo,
                gmx.commit == fp.commit,
            ],
        )
        .left_join(
            tgf,
            [
                tgf.function_goid_h128 == fp.function_goid_h128,
                tgf.repo == fp.repo,
                tgf.commit == fp.commit,
            ],
        )
        .left_join(
            cfg_fn,
            [
                cfg_fn.function_goid_h128 == fp.function_goid_h128,
                cfg_fn.repo == fp.repo,
                cfg_fn.commit == fp.commit,
            ],
        )
        .left_join(
            dfg_fn,
            [
                dfg_fn.function_goid_h128 == fp.function_goid_h128,
                dfg_fn.repo == fp.repo,
                dfg_fn.commit == fp.commit,
            ],
        )
        .left_join(
            mp,
            [
                mp.module == fp.module,
                mp.repo == fp.repo,
                mp.commit == fp.commit,
            ],
        )
        .left_join(
            sm,
            [
                sm.module == fp.module,
                sm.repo == fp.repo,
                sm.commit == fp.commit,
            ],
        )
        .left_join(
            ss,
            [
                ss.subsystem_id == sm.subsystem_id,
                ss.repo == sm.repo,
                ss.commit == sm.commit,
            ],
        )
    )

    return joined.select(
        fp.function_goid_h128,
        fp.repo,
        fp.commit,
        fp.urn,
        fp.rel_path,
        fp.module,
        fp.language,
        fp.kind,
        fp.qualname,
        fp.loc,
        fp.logical_loc,
        fp.cyclomatic_complexity,
        fp.param_count,
        fp.total_params,
        fp.annotated_params,
        fp.return_type,
        fp.typedness_bucket,
        fp.file_typed_ratio,
        fp.coverage_ratio,
        fp.tested,
        fp.tests_touching,
        fp.failing_tests,
        fp.slow_tests,
        fp.created_in_commit,
        fp.created_at_history.name("created_at"),
        fp.last_modified_commit,
        fp.last_modified_at,
        fp.age_days,
        fp.commit_count,
        fp.author_count,
        fp.lines_added,
        fp.lines_deleted,
        fp.churn_score,
        fp.stability_bucket,
        fp.risk_score,
        fp.risk_level,
        fp.is_pure,
        fp.uses_io,
        fp.touches_db,
        fp.uses_time,
        fp.uses_randomness,
        fp.modifies_globals,
        fp.modifies_closure,
        fp.spawns_threads_or_tasks,
        fp.has_transitive_effects,
        fp.purity_confidence,
        fp.param_nullability_json,
        fp.return_nullability,
        fp.has_preconditions,
        fp.has_postconditions,
        fp.has_raises,
        fp.contract_confidence,
        fp.role,
        fp.framework,
        fp.role_confidence,
        fp.role_sources_json,
        fp.tags,
        fp.owners,
        gm.call_fan_in,
        gm.call_fan_out,
        gm.call_in_degree,
        gm.call_out_degree,
        gm.call_pagerank,
        gm.call_betweenness,
        gm.call_closeness,
        gm.call_cycle_member,
        gm.call_cycle_id,
        gm.call_layer,
        gmx.call_betweenness.name("call_betweenness_ext"),
        gmx.call_closeness.name("call_closeness_ext"),
        gmx.call_eigenvector,
        gmx.call_harmonic,
        gmx.call_core_number,
        gmx.call_clustering_coeff,
        gmx.call_triangle_count,
        gmx.call_is_articulation,
        gmx.call_is_bridge_endpoint,
        gmx.call_component_id,
        gmx.call_component_size,
        gmx.call_scc_id,
        gmx.call_scc_size,
        tgf.tests_degree,
        tgf.tests_weighted_degree,
        tgf.tests_degree_centrality,
        tgf.proj_degree.name("tests_co_tested_degree"),
        tgf.proj_weight.name("tests_co_tested_weight"),
        tgf.proj_clustering.name("tests_co_tested_clustering"),
        tgf.proj_betweenness.name("tests_co_tested_betweenness"),
        cfg_fn.cfg_block_count,
        cfg_fn.cfg_edge_count,
        cfg_fn.cfg_has_cycles,
        cfg_fn.cfg_scc_count,
        cfg_fn.cfg_longest_path_len,
        cfg_fn.cfg_avg_shortest_path_len,
        cfg_fn.cfg_branching_factor_mean,
        cfg_fn.cfg_branching_factor_max,
        cfg_fn.cfg_linear_block_fraction,
        cfg_fn.cfg_dom_tree_height,
        cfg_fn.cfg_dominance_frontier_size_mean,
        cfg_fn.cfg_dominance_frontier_size_max,
        cfg_fn.cfg_loop_count,
        cfg_fn.cfg_loop_nesting_depth_max,
        cfg_fn.cfg_bc_betweenness_max,
        cfg_fn.cfg_bc_betweenness_mean,
        cfg_fn.cfg_bc_closeness_mean,
        cfg_fn.cfg_bc_eigenvector_max,
        dfg_fn.dfg_block_count,
        dfg_fn.dfg_edge_count,
        dfg_fn.dfg_phi_edge_count,
        dfg_fn.dfg_symbol_count,
        dfg_fn.dfg_component_count,
        dfg_fn.dfg_scc_count,
        dfg_fn.dfg_has_cycles,
        dfg_fn.dfg_longest_chain_len,
        dfg_fn.dfg_avg_shortest_path_len,
        dfg_fn.dfg_avg_in_degree,
        dfg_fn.dfg_avg_out_degree,
        dfg_fn.dfg_max_in_degree,
        dfg_fn.dfg_max_out_degree,
        dfg_fn.dfg_branchy_block_fraction,
        dfg_fn.dfg_bc_betweenness_max,
        dfg_fn.dfg_bc_betweenness_mean,
        dfg_fn.dfg_bc_eigenvector_max,
        mp.module_coverage_ratio,
        mp.import_fan_in.name("module_import_fan_in"),
        mp.import_fan_out.name("module_import_fan_out"),
        mp.in_cycle.name("module_in_import_cycle"),
        mp.cycle_group.name("module_import_cycle_group"),
        sm.subsystem_id,
        ss.name.name("subsystem_name"),
        ss.risk_level.name("subsystem_risk_level"),
        ss.module_count.name("subsystem_module_count"),
    )


@register_view("docs.v_function_history")
def build_docs_function_history(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_function_history joining profile with history.

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
    fp: it.Table = _table(con, "analytics.function_profile")
    fh: it.Table = _table(con, "analytics.function_history")

    joined = fp.left_join(
        fh,
        [
            fh.repo == fp.repo,
            fh.commit == fp.commit,
            fh.function_goid_h128 == fp.function_goid_h128,
        ],
    )

    return joined.select(
        fp.repo,
        fp.commit,
        fp.function_goid_h128,
        fp.urn,
        fp.rel_path,
        fp.module,
        fp.qualname,
        fh.created_in_commit,
        fh.created_at,
        fh.last_modified_commit,
        fh.last_modified_at,
        fh.age_days,
        fh.commit_count,
        fh.author_count,
        fh.lines_added,
        fh.lines_deleted,
        fh.churn_score,
        fh.stability_bucket,
    )


@register_view("docs.v_function_history_timeseries")
def build_docs_function_history_timeseries(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_function_history_timeseries by filtering history.

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
    h: it.Table = _table(con, "analytics.history_timeseries")

    return h.filter(cast("Any", h.entity_kind == "function")).select(
        h.repo,
        h.entity_stable_id,
        h.commit,
        h.commit_ts,
        h.rel_path,
        h.module,
        h.qualname,
        h.loc,
        h.cyclomatic_complexity,
        h.coverage_ratio,
        h.static_error_count,
        h.typedness_bucket,
        h.risk_score,
        h.risk_level,
        h.bucket_label,
    )


@register_view("docs.v_cfg_block_architecture")
def build_docs_cfg_block_architecture(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_cfg_block_architecture with 4-table join.

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
    cb: it.Table = _table(con, "graph.cfg_blocks")
    g: it.Table = _table(con, "core.goids")
    fp: it.Table = _table(con, "analytics.function_profile")
    bm: it.Table = _table(con, "analytics.cfg_block_metrics")

    joined = (
        cb.inner_join(g, [g.goid_h128 == cb.function_goid_h128])
        .left_join(
            fp,
            [
                fp.function_goid_h128 == cb.function_goid_h128,
                fp.repo == g.repo,
                fp.commit == g.commit,
            ],
        )
        .left_join(
            bm,
            [
                bm.function_goid_h128 == cb.function_goid_h128,
                bm.repo == g.repo,
                bm.commit == g.commit,
                bm.block_idx == cb.block_idx,
            ],
        )
    )

    return joined.select(
        fp.function_goid_h128,
        fp.urn.name("function_urn"),
        fp.repo,
        fp.commit,
        fp.rel_path,
        fp.module,
        fp.kind.name("function_kind"),
        fp.qualname.name("function_qualname"),
        fp.risk_level.name("function_risk_level"),
        fp.risk_score.name("function_risk_score"),
        cb.block_idx,
        cb.block_id,
        cb.label.name("block_label"),
        cb.kind.name("block_kind"),
        cb.file_path.name("block_file_path"),
        cb.start_line.name("block_start_line"),
        cb.end_line.name("block_end_line"),
        cb.in_degree.name("cfg_in_degree"),
        cb.out_degree.name("cfg_out_degree"),
        bm.is_entry,
        bm.is_exit,
        bm.is_branch,
        bm.is_join,
        bm.dom_depth,
        bm.dominates_exit,
        bm.bc_betweenness,
        bm.bc_closeness,
        bm.bc_eigenvector,
        bm.in_loop_scc,
        bm.loop_header,
        bm.loop_nesting_depth,
    )


@register_view("docs.v_dfg_block_architecture")
def build_docs_dfg_block_architecture(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_dfg_block_architecture with 4-table join.

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
    cb: it.Table = _table(con, "graph.cfg_blocks")
    g: it.Table = _table(con, "core.goids")
    fp: it.Table = _table(con, "analytics.function_profile")
    dbm: it.Table = _table(con, "analytics.dfg_block_metrics")

    joined = (
        cb.inner_join(g, [g.goid_h128 == cb.function_goid_h128])
        .left_join(
            fp,
            [
                fp.function_goid_h128 == cb.function_goid_h128,
                fp.repo == g.repo,
                fp.commit == g.commit,
            ],
        )
        .left_join(
            dbm,
            [
                dbm.function_goid_h128 == cb.function_goid_h128,
                dbm.repo == g.repo,
                dbm.commit == g.commit,
                dbm.block_idx == cb.block_idx,
            ],
        )
    )

    return joined.select(
        fp.function_goid_h128,
        fp.urn.name("function_urn"),
        fp.repo,
        fp.commit,
        fp.rel_path,
        fp.module,
        fp.kind.name("function_kind"),
        fp.qualname.name("function_qualname"),
        fp.risk_level.name("function_risk_level"),
        fp.risk_score.name("function_risk_score"),
        cb.block_idx,
        cb.block_id,
        cb.label.name("block_label"),
        cb.kind.name("block_kind"),
        cb.file_path.name("block_file_path"),
        cb.start_line.name("block_start_line"),
        cb.end_line.name("block_end_line"),
        dbm.dfg_in_degree,
        dbm.dfg_out_degree,
        dbm.dfg_phi_in_degree,
        dbm.dfg_phi_out_degree,
        dbm.dfg_bc_betweenness,
        dbm.dfg_bc_closeness,
        dbm.dfg_bc_eigenvector,
        dbm.dfg_in_chain,
        dbm.dfg_in_scc,
    )


# ---------------------------------------------------------------------------
# Module Views
# ---------------------------------------------------------------------------


@register_view("docs.v_module_history_timeseries")
def build_docs_module_history_timeseries(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_module_history_timeseries by filtering history.

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
    h: it.Table = _table(con, "analytics.history_timeseries")

    return h.filter(cast("Any", h.entity_kind == "module")).select(
        h.repo,
        h.entity_stable_id,
        h.commit,
        h.commit_ts,
        h.module,
        h.rel_path,
        h.coverage_ratio,
        h.risk_score,
        h.risk_level,
        h.bucket_label,
    )


@register_view("docs.v_module_architecture_full")
def build_docs_module_architecture_full(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_module_architecture with full 10+ table joins.

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
    gm: it.Table = _table(con, "analytics.graph_metrics_modules")
    m: it.Table = _table(con, "core.modules")
    mp: it.Table = _table(con, "analytics.module_profile")
    gmx: it.Table = _table(con, "analytics.graph_metrics_modules_ext")
    sym: it.Table = _table(con, "analytics.symbol_graph_metrics_modules")
    cfgm: it.Table = _table(con, "analytics.config_graph_metrics_modules")
    sm: it.Table = _table(con, "analytics.subsystem_modules")
    sg: it.Table = _table(con, "analytics.subsystem_graph_metrics")
    sa: it.Table = _table(con, "analytics.subsystem_agreement")

    joined = (
        gm.left_join(m, [m.module == gm.module])
        .left_join(
            mp,
            [
                mp.module == gm.module,
                mp.repo == gm.repo,
                mp.commit == gm.commit,
            ],
        )
        .left_join(
            gmx,
            [
                gmx.module == gm.module,
                gmx.repo == gm.repo,
                gmx.commit == gm.commit,
            ],
        )
        .left_join(
            sym,
            [
                sym.module == gm.module,
                sym.repo == gm.repo,
                sym.commit == gm.commit,
            ],
        )
        .left_join(
            cfgm,
            [
                cfgm.module == gm.module,
                cfgm.repo == gm.repo,
                cfgm.commit == gm.commit,
            ],
        )
        .left_join(
            sm,
            [
                sm.module == gm.module,
                sm.repo == gm.repo,
                sm.commit == gm.commit,
            ],
        )
        .left_join(
            sg,
            [
                sg.subsystem_id == sm.subsystem_id,
                sg.repo == sm.repo,
                sg.commit == sm.commit,
            ],
        )
        .left_join(
            sa,
            [
                sa.module == gm.module,
                sa.repo == gm.repo,
                sa.commit == gm.commit,
            ],
        )
    )

    return joined.select(
        gm.repo,
        gm.commit,
        gm.module,
        m.path.name("rel_path"),
        m.tags,
        m.owners,
        gm.import_fan_in,
        gm.import_fan_out,
        gm.import_in_degree,
        gm.import_out_degree,
        gm.import_pagerank,
        gm.import_betweenness,
        gm.import_closeness,
        gm.import_cycle_member,
        gm.import_cycle_id,
        gm.import_layer,
        gm.symbol_fan_in,
        gm.symbol_fan_out,
        mp.avg_risk_score,
        mp.max_risk_score,
        mp.module_coverage_ratio,
        mp.tested_function_count,
        mp.untested_function_count,
        mp.role,
        mp.role_confidence,
        mp.role_sources_json,
        gmx.import_betweenness.name("import_betweenness_ext"),
        gmx.import_closeness.name("import_closeness_ext"),
        gmx.import_eigenvector,
        gmx.import_harmonic,
        gmx.import_k_core,
        gmx.import_constraint,
        gmx.import_effective_size,
        gmx.import_community_id,
        gmx.import_component_id,
        gmx.import_component_size,
        gmx.import_scc_id,
        gmx.import_scc_size,
        sym.symbol_betweenness,
        sym.symbol_closeness,
        sym.symbol_eigenvector,
        sym.symbol_harmonic,
        sym.symbol_k_core,
        sym.symbol_constraint,
        sym.symbol_effective_size,
        sym.symbol_community_id,
        cfgm.community_id.name("config_community_id"),
        cfgm.degree.name("config_degree"),
        cfgm.weighted_degree.name("config_weighted_degree"),
        cfgm.betweenness.name("config_betweenness"),
        cfgm.closeness.name("config_closeness"),
        sg.import_in_degree.name("subsystem_import_in_degree"),
        sg.import_out_degree.name("subsystem_import_out_degree"),
        sg.import_pagerank.name("subsystem_import_pagerank"),
        sg.import_betweenness.name("subsystem_import_betweenness"),
        sg.import_closeness.name("subsystem_import_closeness"),
        sg.import_layer.name("subsystem_import_layer"),
        sa.import_community_id.name("subsystem_agreed_import_community_id"),
        sa.agrees.name("subsystem_import_agreement"),
        gm.created_at,
    )


@register_view("docs.v_entrypoints")
def build_docs_entrypoints(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_entrypoints (simple passthrough).

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
    return _table(con, "analytics.entrypoints")


@register_view("docs.v_external_dependencies")
def build_docs_external_dependencies(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_external_dependencies (simple passthrough).

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
    return _table(con, "analytics.external_dependencies")


@register_view("docs.v_external_dependency_calls")
def build_docs_external_dependency_calls(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_external_dependency_calls (simple passthrough).

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
    return _table(con, "analytics.external_dependency_calls")


# ---------------------------------------------------------------------------
# Data Model Views
# ---------------------------------------------------------------------------


@register_view("docs.v_data_models")
def build_docs_data_models(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_data_models with nested fields and relationships JSON.

    This view uses correlated subqueries with DuckDB's list() and struct_pack()
    to build nested JSON arrays for fields and relationships. Since Ibis doesn't
    natively support correlated subqueries, we use Ibis's sql() capability.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    # Use raw SQL for this complex view with correlated subqueries
    # The struct_pack and list functions are DuckDB-specific
    sql_expr = """
        SELECT
            dm.repo,
            dm.commit,
            dm.model_id,
            dm.goid_h128,
            dm.model_name,
            dm.module,
            dm.rel_path,
            dm.model_kind,
            coalesce(dm.base_classes_json, '[]') AS base_classes_json,
            (
                SELECT to_json(
                           coalesce(
                               list(
                                   struct_pack(
                                       name := f.field_name,
                                       type := f.field_type,
                                       required := f.required,
                                       has_default := f.has_default,
                                       default_expr := f.default_expr,
                                       constraints := f.constraints_json,
                                       source := f.source,
                                       lineno := f.lineno
                                   )
                                   ORDER BY f.field_name
                               ),
                               []
                           )
                       )
                FROM analytics.data_model_fields f
                WHERE f.repo = dm.repo AND f.commit = dm.commit AND f.model_id = dm.model_id
            ) AS fields,
            (
                SELECT to_json(
                           coalesce(
                               list(
                                   struct_pack(
                                       field := r.field_name,
                                       target_model_id := r.target_model_id,
                                       target_model_name := r.target_model_name,
                                       target_module := r.target_module,
                                       multiplicity := r.multiplicity,
                                       kind := r.relationship_kind,
                                       via := r.via,
                                       rel_path := r.rel_path,
                                       lineno := r.lineno,
                                       evidence := r.evidence_json
                                   )
                                   ORDER BY r.field_name
                               ),
                               []
                           )
                       )
                FROM analytics.data_model_relationships r
                WHERE r.repo = dm.repo
                  AND r.commit = dm.commit
                  AND r.source_model_id = dm.model_id
            ) AS relationships,
            dm.doc_short,
            dm.doc_long,
            dm.created_at
        FROM analytics.data_models dm
    """
    return ibis_gw.con.sql(sql_expr)


@register_view("docs.v_data_models_normalized")
def build_docs_data_models_normalized(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_data_models_normalized with nested struct arrays.

    Similar to v_data_models but returns DuckDB list types (arrays of structs)
    instead of JSON. This is used for efficient programmatic access to the
    nested field and relationship data.

    Parameters
    ----------
    ibis_gw
        Ibis gateway for table access.

    Returns
    -------
    it.Table
        Ibis table expression for the view.
    """
    # Use raw SQL for this complex view with correlated subqueries
    sql_expr = """
        SELECT
            dm.repo,
            dm.commit,
            dm.model_id,
            dm.goid_h128,
            dm.model_name,
            dm.module,
            dm.rel_path,
            dm.model_kind,
            coalesce(dm.base_classes_json, '[]') AS base_classes_json,
            (SELECT coalesce(list(
                        struct_pack(
                            field_name := f.field_name,
                            field_type := f.field_type,
                            required := f.required,
                            has_default := f.has_default,
                            default_expr := f.default_expr,
                            constraints := f.constraints_json,
                            source := f.source,
                            rel_path := f.rel_path,
                            lineno := f.lineno,
                            created_at := f.created_at
                        )
                        ORDER BY f.field_name
                    )
                    , []
                    )
             FROM analytics.data_model_fields f
             WHERE f.repo = dm.repo AND f.commit = dm.commit AND f.model_id = dm.model_id
            ) AS fields,
            (SELECT coalesce(list(
                        struct_pack(
                            field_name := r.field_name,
                            target_model_id := r.target_model_id,
                            target_module := r.target_module,
                            target_model_name := r.target_model_name,
                            relationship_kind := r.relationship_kind,
                            multiplicity := r.multiplicity,
                            via := r.via,
                            evidence := r.evidence_json,
                            rel_path := r.rel_path,
                            lineno := r.lineno,
                            created_at := r.created_at
                        )
                        ORDER BY r.field_name
                    )
                    , []
                    )
             FROM analytics.data_model_relationships r
             WHERE r.repo = dm.repo AND r.commit = dm.commit AND r.source_model_id = dm.model_id
            ) AS relationships,
            dm.doc_short,
            dm.doc_long,
            dm.created_at
        FROM analytics.data_models dm
    """
    return ibis_gw.con.sql(sql_expr)


@register_view("docs.v_data_model_fields")
def build_docs_data_model_fields(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_data_model_fields (simple passthrough).

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
    return _table(con, "analytics.data_model_fields")


@register_view("docs.v_data_model_relationships")
def build_docs_data_model_relationships(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_data_model_relationships (simple passthrough).

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
    return _table(con, "analytics.data_model_relationships")


@register_view("docs.v_data_model_usage")
def build_docs_data_model_usage(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_data_model_usage joining usage with models and profiles.

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
    u: it.Table = _table(con, "analytics.data_model_usage")
    dm: it.Table = _table(con, "analytics.data_models")
    fp: it.Table = _table(con, "analytics.function_profile")

    joined = u.left_join(
        dm,
        [
            dm.repo == u.repo,
            dm.commit == u.commit,
            dm.model_id == u.model_id,
        ],
    ).left_join(
        fp,
        [
            fp.repo == u.repo,
            fp.commit == u.commit,
            fp.function_goid_h128 == u.function_goid_h128,
        ],
    )

    return joined.select(
        u.repo,
        u.commit,
        u.model_id,
        dm.model_name,
        dm.model_kind,
        u.function_goid_h128,
        fp.qualname.name("function_qualname"),
        fp.rel_path.name("function_rel_path"),
        fp.risk_score,
        fp.coverage_ratio,
        u.usage_kinds_json,
        u.context_json,
        u.evidence_json,
        u.created_at,
    )


@register_view("docs.v_config_data_flow")
def build_docs_config_data_flow(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_config_data_flow joining config flow with profiles.

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
    c: it.Table = _table(con, "analytics.config_data_flow")
    fp: it.Table = _table(con, "analytics.function_profile")

    joined = c.left_join(
        fp,
        [
            fp.repo == c.repo,
            fp.commit == c.commit,
            fp.function_goid_h128 == c.function_goid_h128,
        ],
    )

    return joined.select(
        c.repo,
        c.commit,
        c.config_key,
        c.config_path,
        c.function_goid_h128,
        fp.qualname.name("function_qualname"),
        fp.rel_path.name("function_rel_path"),
        fp.risk_score,
        fp.coverage_ratio,
        c.usage_kind,
        c.evidence_json,
        c.call_chain_id,
        c.call_chain_json,
        c.created_at,
    )


# ---------------------------------------------------------------------------
# Subsystem Views (additional)
# ---------------------------------------------------------------------------


@register_view("docs.v_module_with_subsystem")
def build_docs_module_with_subsystem(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_module_with_subsystem joining modules with subsystems.

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
    sm: it.Table = _table(con, "analytics.subsystem_modules")
    subs: it.Table = _table(con, "analytics.subsystems")
    m: it.Table = _table(con, "core.modules")
    gm: it.Table = _table(con, "analytics.graph_metrics_modules")

    joined = (
        sm.left_join(
            m,
            [
                m.repo == sm.repo,
                m.commit == sm.commit,
                m.module == sm.module,
            ],
        )
        .left_join(
            gm,
            [
                gm.repo == sm.repo,
                gm.commit == sm.commit,
                gm.module == sm.module,
            ],
        )
        .left_join(
            subs,
            [
                subs.repo == sm.repo,
                subs.commit == sm.commit,
                subs.subsystem_id == sm.subsystem_id,
            ],
        )
    )

    return joined.select(
        sm.repo,
        sm.commit,
        sm.subsystem_id,
        subs.name.name("subsystem_name"),
        sm.module,
        sm.role,
        m.path.name("rel_path"),
        m.tags,
        m.owners,
        gm.import_fan_in,
        gm.import_fan_out,
        gm.symbol_fan_in,
        gm.symbol_fan_out,
        subs.risk_level,
        subs.avg_risk_score,
        subs.max_risk_score,
    )


@register_view("docs.v_subsystem_agreement")
def build_docs_subsystem_agreement(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_subsystem_agreement (simple passthrough).

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
    return _table(con, "analytics.subsystem_agreement")


# ---------------------------------------------------------------------------
# Test Views
# ---------------------------------------------------------------------------


@register_view("docs.v_test_to_function")
def build_docs_test_to_function(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_test_to_function joining edges with catalog and risk.

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
    e: it.Table = _table(con, "analytics.test_coverage_edges")
    tc: it.Table = _table(con, "analytics.test_catalog")
    rf: it.Table = _table(con, "analytics.goid_risk_factors")

    joined = e.left_join(
        tc,
        [
            e.test_id == tc.test_id,
            e.repo == tc.repo,
            e.commit == tc.commit,
        ],
    ).left_join(
        rf,
        [
            rf.function_goid_h128 == e.function_goid_h128,
            rf.repo == e.repo,
            rf.commit == e.commit,
        ],
    )

    return joined.select(
        e.test_id,
        tc.test_goid_h128,
        tc.urn.name("test_urn"),
        tc.repo.name("test_repo"),
        tc.commit.name("test_commit"),
        tc.rel_path.name("test_rel_path"),
        tc.qualname.name("test_qualname"),
        tc.kind.name("test_kind"),
        tc.status.name("test_status"),
        tc.duration_ms,
        tc.markers,
        tc.parametrized,
        tc.flaky,
        e.function_goid_h128,
        rf.urn.name("function_urn"),
        rf.rel_path.name("function_rel_path"),
        rf.qualname.name("function_qualname"),
        rf.language.name("function_language"),
        rf.kind.name("function_kind"),
        e.covered_lines,
        e.executable_lines,
        e.coverage_ratio,
        e.last_status.name("edge_last_status"),
        rf.risk_score.name("function_risk_score"),
        rf.risk_level.name("function_risk_level"),
        e.created_at.name("edge_created_at"),
    )


@register_view("docs.v_test_architecture")
def build_docs_test_architecture(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_test_architecture joining profile with behavioral coverage.

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
    p: it.Table = _table(con, "analytics.test_profile")
    b: it.Table = _table(con, "analytics.behavioral_coverage")

    joined = p.left_join(
        b,
        [
            b.repo == p.repo,
            b.commit == p.commit,
            b.test_id == p.test_id,
        ],
    )

    return joined.select(
        p.repo,
        p.commit,
        p.test_id,
        p.test_goid_h128,
        p.urn,
        p.rel_path,
        p.module,
        p.qualname,
        p.language,
        p.kind,
        p.status,
        p.duration_ms,
        p.markers,
        p.flaky,
        p.flakiness_score,
        p.importance_score,
        p.functions_covered,
        p.functions_covered_count,
        p.primary_function_goids,
        p.subsystems_covered,
        p.subsystems_covered_count,
        p.primary_subsystem_id,
        p.assert_count,
        p.raise_count,
        p.uses_parametrize,
        p.uses_fixtures,
        p.io_bound,
        p.uses_network,
        p.uses_db,
        p.uses_filesystem,
        p.uses_subprocess,
        p.tg_degree,
        p.tg_weighted_degree,
        p.tg_proj_degree,
        p.tg_proj_weight,
        p.tg_proj_clustering,
        p.tg_proj_betweenness,
        b.behavior_tags,
        b.tag_source,
        b.heuristic_version,
        b.llm_model,
        b.llm_run_id,
        p.created_at,
    )


@register_view("docs.v_behavioral_classification_input")
def build_docs_behavioral_classification_input(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_behavioral_classification_input for LLM classification.

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
    p: it.Table = _table(con, "analytics.test_profile")
    b: it.Table = _table(con, "analytics.behavioral_coverage")

    joined = p.left_join(
        b,
        [
            b.repo == p.repo,
            b.commit == p.commit,
            b.test_id == p.test_id,
        ],
    )

    return joined.select(
        p.repo,
        p.commit,
        p.test_id,
        p.rel_path,
        p.qualname,
        p.markers,
        p.functions_covered,
        p.subsystems_covered,
        p.assert_count,
        p.raise_count,
        p.status,
        p.duration_ms,
        p.flaky,
        b.behavior_tags,
        b.tag_source,
        b.heuristic_version,
    )


# ---------------------------------------------------------------------------
# Graph Views
# ---------------------------------------------------------------------------


@register_view("docs.v_symbol_module_graph")
def build_docs_symbol_module_graph(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_symbol_module_graph (simple passthrough).

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
    return _table(con, "analytics.symbol_graph_metrics_modules")


@register_view("docs.v_validation_summary")
def build_docs_validation_summary(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_validation_summary as union of function and graph validation.

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
    fv: it.Table = _table(con, "analytics.function_validation")
    gv: it.Table = _table(con, "analytics.graph_validation")

    func_part = fv.select(
        ibis.literal("function").name("domain"),
        fv.repo,
        fv.commit,
        fv.function_goid_h128.cast("string").name("entity_id"),
        fv.issue,
        fv.detail,
    )

    graph_part = gv.select(
        ibis.literal("graph").name("domain"),
        gv.repo,
        gv.commit,
        gv.entity_id.cast("string").name("entity_id"),
        gv.issue,
        gv.detail,
    )

    return func_part.union(graph_part)


# ---------------------------------------------------------------------------
# IDE Views
# ---------------------------------------------------------------------------


@register_view("docs.v_ide_hints")
def build_docs_ide_hints(ibis_gw: IbisGateway) -> it.Table:
    """Build docs.v_ide_hints joining modules with architecture and subsystems.

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
    m: it.Table = _table(con, "core.modules")
    mp: it.Table = _table(con, "analytics.module_profile")
    gm: it.Table = _table(con, "analytics.graph_metrics_modules")
    sm: it.Table = _table(con, "analytics.subsystem_modules")
    subs: it.Table = _table(con, "analytics.subsystems")

    joined = (
        m.left_join(
            gm,
            [
                gm.repo == m.repo,
                gm.commit == m.commit,
                gm.module == m.module,
            ],
        )
        .left_join(
            mp,
            [
                mp.repo == m.repo,
                mp.commit == m.commit,
                mp.module == m.module,
            ],
        )
        .left_join(
            sm,
            [
                sm.repo == m.repo,
                sm.commit == m.commit,
                sm.module == m.module,
            ],
        )
        .left_join(
            subs,
            [
                subs.repo == sm.repo,
                subs.commit == sm.commit,
                subs.subsystem_id == sm.subsystem_id,
            ],
        )
    )

    return joined.select(
        m.repo,
        m.commit,
        m.path.name("rel_path"),
        m.module,
        gm.import_fan_in,
        gm.import_fan_out,
        gm.symbol_fan_in,
        gm.symbol_fan_out,
        mp.avg_risk_score.name("module_avg_risk_score"),
        mp.max_risk_score.name("module_max_risk_score"),
        mp.module_coverage_ratio,
        mp.tested_function_count,
        mp.untested_function_count,
        m.tags,
        m.owners,
        subs.subsystem_id,
        subs.name.name("subsystem_name"),
        subs.description.name("subsystem_description"),
        sm.role.name("subsystem_role"),
        subs.risk_level.name("subsystem_risk_level"),
        subs.module_count.name("subsystem_module_count"),
        subs.entrypoints_json.name("subsystem_entrypoints"),
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
    """Create supplementary analytics.* and core.* Ibis views.

    This function creates views in the analytics.* and core.* namespaces that
    provide additional computed aggregations. These supplement the docs.* views
    which are created separately via VIEW_BUILDERS registry in views/__init__.py.

    Note: All docs.* views are now created via create_all_views() using the
    VIEW_BUILDERS registry. This function only creates the following:
    - analytics.v_function_summary
    - analytics.v_callgraph_degree
    - core.v_goid_crosswalk_*
    - analytics.v_function_hotspots
    - analytics.v_import_degree
    """
    create_function_summary_view(gateway)  # analytics.v_function_summary
    create_callgraph_degree_view(gateway)  # analytics.v_callgraph_degree
    create_goid_crosswalk_views(gateway)  # core.v_goid_crosswalk_*
    create_function_hotspots_view(gateway)  # analytics.v_function_hotspots
    create_import_graph_degree_view(gateway)  # analytics.v_import_degree
