"""DuckDB view definitions for docs.* surfaces."""

from __future__ import annotations

import duckdb


def create_all_views(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create or replace AI-facing views in the `docs` schema.

    These views back both the FastAPI service and MCP tools.
    """
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
            rf.created_at
        FROM analytics.goid_risk_factors rf
        LEFT JOIN analytics.function_metrics fm
          ON fm.function_goid_h128 = rf.function_goid_h128
         AND fm.repo = rf.repo
         AND fm.commit = rf.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_call_graph_enriched AS
        SELECT
            e.caller_goid_h128,
            gc.repo         AS caller_repo,
            gc.commit       AS caller_commit,
            gc.urn           AS caller_urn,
            gc.rel_path      AS caller_rel_path,
            gc.qualname      AS caller_qualname,
            rc.risk_level    AS caller_risk_level,
            rc.risk_score    AS caller_risk_score,
            e.callee_goid_h128,
            gcallee.repo     AS callee_repo,
            gcallee.commit   AS callee_commit,
            gcallee.urn      AS callee_urn,
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
         AND gc.repo = e.repo
         AND gc.commit = e.commit
        LEFT JOIN core.goids gcallee
          ON gcallee.goid_h128 = e.callee_goid_h128
         AND gcallee.repo = e.repo
         AND gcallee.commit = e.commit
        LEFT JOIN analytics.goid_risk_factors rc
          ON rc.function_goid_h128 = e.caller_goid_h128
         AND rc.repo = e.repo
         AND rc.commit = e.commit
        LEFT JOIN analytics.goid_risk_factors rcallee
          ON rcallee.function_goid_h128 = e.callee_goid_h128
         AND rcallee.repo = e.repo
         AND rcallee.commit = e.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_architecture AS
        SELECT
            gm.function_goid_h128,
            gm.repo,
            gm.commit,
            rf.urn,
            rf.rel_path,
            rf.qualname,
            rf.language,
            rf.kind,
            rf.risk_score,
            rf.risk_level,
            rf.tags,
            rf.owners,
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
            gm.created_at
        FROM analytics.graph_metrics_functions gm
        LEFT JOIN analytics.goid_risk_factors rf
          ON rf.function_goid_h128 = gm.function_goid_h128
         AND rf.repo = gm.repo
         AND rf.commit = gm.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_module_architecture AS
        SELECT
            gm.repo,
            gm.commit,
            gm.module,
            m.path          AS rel_path,
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
            gm.created_at
        FROM analytics.graph_metrics_modules gm
        LEFT JOIN core.modules m
          ON m.module = gm.module
        LEFT JOIN analytics.module_profile mp
          ON mp.module = gm.module
         AND mp.repo = gm.repo
         AND mp.commit = gm.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_subsystem_summary AS
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
            s.created_at
        FROM analytics.subsystems s;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_module_with_subsystem AS
        SELECT
            sm.repo,
            sm.commit,
            sm.subsystem_id,
            subs.name              AS subsystem_name,
            sm.module,
            sm.role,
            ma.rel_path,
            ma.tags,
            ma.owners,
            ma.import_fan_in,
            ma.import_fan_out,
            ma.symbol_fan_in,
            ma.symbol_fan_out,
            subs.risk_level,
            subs.avg_risk_score,
            subs.max_risk_score
        FROM analytics.subsystem_modules sm
        LEFT JOIN docs.v_module_architecture ma
          ON ma.repo = sm.repo
         AND ma.commit = sm.commit
         AND ma.module = sm.module
        LEFT JOIN analytics.subsystems subs
          ON subs.repo = sm.repo
         AND subs.commit = sm.commit
         AND subs.subsystem_id = sm.subsystem_id;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_ide_hints AS
        SELECT
            m.repo,
            m.commit,
            m.path                    AS rel_path,
            m.module,
            ma.import_fan_in,
            ma.import_fan_out,
            ma.symbol_fan_in,
            ma.symbol_fan_out,
            ma.avg_risk_score         AS module_avg_risk_score,
            ma.max_risk_score         AS module_max_risk_score,
            ma.module_coverage_ratio,
            ma.tested_function_count,
            ma.untested_function_count,
            m.tags,
            m.owners,
            subs.subsystem_id,
            subs.name                 AS subsystem_name,
            subs.description          AS subsystem_description,
            sm.role                   AS subsystem_role,
            subs.risk_level           AS subsystem_risk_level,
            subs.module_count         AS subsystem_module_count,
            subs.entrypoints_json     AS subsystem_entrypoints
        FROM core.modules m
        LEFT JOIN docs.v_module_architecture ma
          ON ma.repo = m.repo
         AND ma.commit = m.commit
         AND ma.module = m.module
        LEFT JOIN analytics.subsystem_modules sm
          ON sm.repo = m.repo
         AND sm.commit = m.commit
         AND sm.module = m.module
        LEFT JOIN analytics.subsystems subs
          ON subs.repo = sm.repo
         AND subs.commit = sm.commit
         AND subs.subsystem_id = sm.subsystem_id;
        """
    )

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
         AND e.repo = tc.repo
         AND e.commit = tc.commit
        LEFT JOIN analytics.goid_risk_factors rf
          ON rf.function_goid_h128 = e.function_goid_h128
         AND rf.repo = e.repo
         AND rf.commit = e.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_file_summary AS
        WITH per_file_risk AS (
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
          ON r.rel_path = m.path
         AND r.repo = m.repo
         AND r.commit = m.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_profile AS
        SELECT *
        FROM analytics.function_profile;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_file_profile AS
        SELECT *
        FROM analytics.file_profile;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_module_profile AS
        SELECT *
        FROM analytics.module_profile;
        """
    )
