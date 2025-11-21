# src/codeintel_core/config/views.py

from __future__ import annotations

import duckdb


def create_all_views(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create / replace AI-facing views in the `docs` schema.

    These views are the main contract for agents and external tools.
    """

    # ----------------------------------------------------------------------
    # docs.v_function_summary
    # One row per function GOID with risk, coverage, and basic metrics.
    # ----------------------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_summary AS
        SELECT
            rf.function_goid_h128,
            rf.urn,
            rf.repo,
            rf.commit,
            rf.rel_path,
            rf.language,
            rf.kind,
            rf.qualname,

            -- core metrics
            rf.loc,
            rf.logical_loc,
            rf.cyclomatic_complexity,
            rf.complexity_bucket,

            -- function signature traits (from function_metrics)
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

            -- typedness
            rf.typedness_bucket,
            rf.typedness_source,

            -- file-level & diagnostics
            rf.hotspot_score,
            rf.file_typed_ratio,
            rf.static_error_count,
            rf.has_static_errors,

            -- coverage & tests
            rf.executable_lines,
            rf.covered_lines,
            rf.coverage_ratio,
            rf.tested,
            rf.test_count,
            rf.failing_test_count,
            rf.last_test_status,

            -- overall risk
            rf.risk_score,
            rf.risk_level,

            -- ownership & tags
            rf.tags,
            rf.owners,

            rf.created_at
        FROM analytics.goid_risk_factors rf
        LEFT JOIN analytics.function_metrics fm
          ON fm.function_goid_h128 = rf.function_goid_h128;
        """
    )

    # ----------------------------------------------------------------------
    # docs.v_call_graph_enriched
    # Call graph edges augmented with human-readable names and risk.
    # ----------------------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_call_graph_enriched AS
        SELECT
            e.caller_goid_h128,
            gc.urn          AS caller_urn,
            gc.rel_path     AS caller_rel_path,
            gc.qualname     AS caller_qualname,
            rc.risk_level   AS caller_risk_level,
            rc.risk_score   AS caller_risk_score,

            e.callee_goid_h128,
            gcallee.urn     AS callee_urn,
            gcallee.rel_path AS callee_rel_path,
            gcallee.qualname AS callee_qualname,
            rcallee.risk_level AS callee_risk_level,
            rcallee.risk_score AS callee_risk_score,

            e.callsite_path,
            e.callsite_line,
            e.callsite_col,
            e.language,
            e.kind,
            e.resolved_via,
            e.confidence,
            e.evidence_json
        FROM graph.call_graph_edges e
        LEFT JOIN core.goids gc
          ON gc.goid_h128 = e.caller_goid_h128
        LEFT JOIN core.goids gcallee
          ON gcallee.goid_h128 = e.callee_goid_h128
        LEFT JOIN analytics.goid_risk_factors rc
          ON rc.function_goid_h128 = e.caller_goid_h128
        LEFT JOIN analytics.goid_risk_factors rcallee
          ON rcallee.function_goid_h128 = e.callee_goid_h128;
        """
    )

    # ----------------------------------------------------------------------
    # docs.v_test_to_function
    # Tests ↔ functions, with per-edge coverage and function risk.
    # ----------------------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_test_to_function AS
        SELECT
            e.test_id,
            tc.test_goid_h128,
            tc.urn              AS test_urn,
            tc.repo             AS test_repo,
            tc.commit           AS test_commit,
            tc.rel_path         AS test_rel_path,
            tc.qualname         AS test_qualname,
            tc.kind             AS test_kind,
            tc.status           AS test_status,
            tc.duration_ms,
            tc.markers,
            tc.parametrized,
            tc.flaky,

            e.function_goid_h128,
            rf.urn              AS function_urn,
            rf.rel_path         AS function_rel_path,
            rf.qualname         AS function_qualname,
            rf.language         AS function_language,
            rf.kind             AS function_kind,

            e.covered_lines,
            e.executable_lines,
            e.coverage_ratio,
            e.last_status       AS edge_last_status,

            rf.risk_score       AS function_risk_score,
            rf.risk_level       AS function_risk_level,

            e.created_at        AS edge_created_at
        FROM analytics.test_coverage_edges e
        LEFT JOIN analytics.test_catalog tc
          ON e.test_id = tc.test_id
        LEFT JOIN analytics.goid_risk_factors rf
          ON rf.function_goid_h128 = e.function_goid_h128;
        """
    )

    # ----------------------------------------------------------------------
    # docs.v_file_summary
    # Per-file view combining AST metrics, hotspots, typedness, diagnostics,
    # and aggregated function risk.
    # ----------------------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_file_summary AS
        WITH per_file_risk AS (
            SELECT
                rel_path,
                COUNT(*) AS function_count,
                SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) AS high_risk_functions,
                SUM(CASE WHEN risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_risk_functions,
                SUM(CASE WHEN risk_level = 'low' THEN 1 ELSE 0 END) AS low_risk_functions,
                MAX(risk_score) AS max_risk_score
            FROM analytics.goid_risk_factors
            GROUP BY rel_path
        )
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

            r.function_count,
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
        LEFT JOIN per_file_risk r
          ON r.rel_path = m.path;
        """
    )
