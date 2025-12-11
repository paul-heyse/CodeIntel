"""Ibis-defined views for analytics and core datasets."""

from __future__ import annotations

from typing import cast

import ibis
import ibis.expr.types as it

from codeintel.storage.gateway.protocol import StorageGateway

__all__ = [
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

    loc_col = cast("it.NumericValue", joined["loc"])
    cyclomatic_complexity = cast("it.NumericValue", joined.cyclomatic_complexity)
    small_loc = loc_col <= ibis.literal(CALLGRAPH_LOC_SMALL)
    medium_loc = loc_col <= ibis.literal(CALLGRAPH_LOC_MEDIUM)
    low_complexity = cyclomatic_complexity <= ibis.literal(COMPLEXITY_LOW_MAX)
    medium_complexity = cyclomatic_complexity <= ibis.literal(COMPLEXITY_MEDIUM_MAX)
    loc_bucket = (
        ibis.case().when(small_loc, "small").when(medium_loc, "medium").else_("large").end()
    )
    complexity_band = (
        ibis.case()
        .when(low_complexity, "low")
        .when(medium_complexity, "medium")
        .else_("high")
        .end()
    )

    enriched = joined.mutate(
        loc_bucket=loc_bucket,
        complexity_band=complexity_band,
    )
    summary = enriched.select(
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

    con.create_view("analytics.v_function_summary", summary, overwrite=True)


def create_docs_function_summary_view(gateway: StorageGateway) -> None:
    """Create docs.v_function_summary derived from risk factors and metrics."""
    con = gateway.ibis.con
    rf: it.Table = con.table("analytics.goid_risk_factors")
    fm: it.Table = con.table("analytics.function_metrics")

    joined = rf.left_join(
        fm,
        [
            rf.function_goid_h128 == fm.function_goid_h128,
            rf.repo == fm.repo,
            rf.commit == fm.commit,
        ],
    )
    summary = joined.select(
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
    con.create_view("docs.v_function_summary", summary, overwrite=True)


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

    repo_value = ibis.coalesce(out_degree.repo, in_degree.repo)
    commit_value = ibis.coalesce(out_degree.commit, in_degree.commit)
    function_goid = ibis.coalesce(out_degree.caller_goid_h128, in_degree.callee_goid_h128)
    call_out_degree = ibis.coalesce(out_degree.call_out_degree, ibis.literal(0))
    call_in_degree = ibis.coalesce(in_degree.call_in_degree, ibis.literal(0))

    degree_view = joined.select(
        repo=repo_value,
        commit=commit_value,
        function_goid_h128=function_goid,
        call_out_degree=call_out_degree,
        call_in_degree=call_in_degree,
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
    create_docs_function_summary_view(gateway)
    create_function_summary_view(gateway)
    create_callgraph_degree_view(gateway)
    create_call_graph_enriched_view(gateway)
    create_goid_crosswalk_views(gateway)
    create_function_hotspots_view(gateway)
    create_import_graph_degree_view(gateway)
    create_docs_file_summary_view(gateway)
    create_docs_module_architecture_view(gateway)
    create_docs_subsystem_summary_view(gateway)
    create_docs_subsystem_profile_view(gateway)
    create_docs_subsystem_coverage_view(gateway)


def create_function_hotspots_view(gateway: StorageGateway) -> None:
    """
    Create analytics.v_function_hotspots using goid risk factors.

    Normalizes hotspot_score to [0,1] for relative comparisons.
    """
    con = gateway.ibis.con
    rf: it.Table = con.table("analytics.goid_risk_factors")
    min_score = rf.hotspot_score.min()
    max_score = rf.hotspot_score.max()
    score_range = max_score.cast("float64") - min_score.cast("float64")
    normalized_score = (
        ibis.case()
        .when(score_range == 0, 0.0)
        .else_((rf.hotspot_score.cast("float64") - min_score.cast("float64")) / score_range)
        .end()
    )
    hotspots = rf.mutate(
        hotspot_normalized=normalized_score,
    ).select(
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
    con.create_view("analytics.v_function_hotspots", hotspots, overwrite=True)


def create_call_graph_enriched_view(gateway: StorageGateway) -> None:
    """Create docs.v_call_graph_enriched to align with the SQL definition."""
    con = gateway.ibis.con
    edges: it.Table = con.table("graph.call_graph_edges")
    goids: it.Table = con.table("core.goids")
    risk: it.Table = con.table("analytics.goid_risk_factors")

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

    enriched = joined.select(
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
    con.create_view("docs.v_call_graph_enriched", enriched, overwrite=True)


def create_import_graph_degree_view(gateway: StorageGateway) -> None:
    """Create graph.v_import_graph_degree aggregating import edge degrees."""
    con = gateway.ibis.con
    edges: it.Table = con.table("graph.import_graph_edges")

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
    degree_view = joined.select(
        repo=ibis.coalesce(out_degree.repo, in_degree.repo),
        commit=ibis.coalesce(out_degree.commit, in_degree.commit),
        module=ibis.coalesce(out_degree.src_module, in_degree.dst_module),
        import_out_degree=ibis.coalesce(out_degree.import_out_degree, ibis.literal(0)),
        import_in_degree=ibis.coalesce(in_degree.import_in_degree, ibis.literal(0)),
    )
    con.create_view("graph.v_import_graph_degree", degree_view, overwrite=True)


def create_docs_file_summary_view(gateway: StorageGateway) -> None:
    """
    Create docs.v_file_summary aggregating per-file statistics.

    Combines file profiles with ownership and module information.
    """
    con = gateway.ibis.con
    fp: it.Table = con.table("analytics.file_profile")
    modules: it.Table = con.table("core.modules")

    joined = fp.left_join(
        modules,
        [
            fp.repo == modules.repo,
            fp.commit == modules.commit,
            fp.rel_path == modules.path,
        ],
    )

    summary = joined.select(
        fp.repo,
        fp.commit,
        fp.rel_path,
        modules.module.name("module"),
        fp.language,
        fp.function_count,
        fp.class_count,
        fp.loc,
        fp.complexity,
        fp.avg_risk_score,
        fp.max_risk_score,
        fp.high_risk_function_count,
        fp.coverage_ratio,
        fp.typed_ratio,
        fp.hotspot_score,
        fp.static_error_count,
        modules.tags,
        modules.owners,
    )
    con.create_view("docs.v_file_summary", summary, overwrite=True)


def create_docs_module_architecture_view(gateway: StorageGateway) -> None:
    """
    Create docs.v_module_architecture combining module metrics with subsystem.

    Provides an architectural view of each module with graph metrics.
    """
    con = gateway.ibis.con
    modules: it.Table = con.table("core.modules")
    graph_metrics: it.Table = con.table("analytics.graph_metrics_modules")
    subsystem_modules: it.Table = con.table("analytics.subsystem_modules")
    subsystems: it.Table = con.table("analytics.subsystems")

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

    architecture = joined.select(
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
    con.create_view("docs.v_module_architecture", architecture, overwrite=True)


def create_docs_subsystem_summary_view(gateway: StorageGateway) -> None:
    """
    Create docs.v_subsystem_summary combining subsystem with profile cache.

    Provides an overview of each subsystem's structure and risk profile.
    """
    con = gateway.ibis.con
    subsystems: it.Table = con.table("analytics.subsystems")
    profile: it.Table = con.table("analytics.subsystem_profile_cache")

    joined = subsystems.left_join(
        profile,
        [
            subsystems.repo == profile.repo,
            subsystems.commit == profile.commit,
            subsystems.subsystem_id == profile.subsystem_id,
        ],
    )

    summary = joined.select(
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
    con.create_view("docs.v_subsystem_summary", summary, overwrite=True)


def create_docs_subsystem_profile_view(gateway: StorageGateway) -> None:
    """
    Create docs.v_subsystem_profile with full profile and graph metrics.

    Extended subsystem view including graph-level metrics.
    """
    con = gateway.ibis.con
    subsystems: it.Table = con.table("analytics.subsystems")
    profile: it.Table = con.table("analytics.subsystem_profile_cache")
    graph_metrics: it.Table = con.table("analytics.subsystem_graph_metrics")

    joined = (
        subsystems.left_join(
            profile,
            [
                subsystems.repo == profile.repo,
                subsystems.commit == profile.commit,
                subsystems.subsystem_id == profile.subsystem_id,
            ],
        )
        .left_join(
            graph_metrics,
            [
                subsystems.repo == graph_metrics.repo,
                subsystems.commit == graph_metrics.commit,
                subsystems.subsystem_id == graph_metrics.subsystem_id,
            ],
        )
    )

    profile_view = joined.select(
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
    con.create_view("docs.v_subsystem_profile", profile_view, overwrite=True)


def create_docs_subsystem_coverage_view(gateway: StorageGateway) -> None:
    """
    Create docs.v_subsystem_coverage combining subsystem with coverage data.

    Provides test coverage metrics per subsystem.
    """
    con = gateway.ibis.con
    subsystems: it.Table = con.table("analytics.subsystems")
    profile: it.Table = con.table("analytics.subsystem_profile_cache")
    coverage: it.Table = con.table("analytics.subsystem_coverage_cache")

    joined = (
        subsystems.left_join(
            profile,
            [
                subsystems.repo == profile.repo,
                subsystems.commit == profile.commit,
                subsystems.subsystem_id == profile.subsystem_id,
            ],
        )
        .left_join(
            coverage,
            [
                subsystems.repo == coverage.repo,
                subsystems.commit == coverage.commit,
                subsystems.subsystem_id == coverage.subsystem_id,
            ],
        )
    )

    coverage_view = joined.select(
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
    con.create_view("docs.v_subsystem_coverage", coverage_view, overwrite=True)
