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
            fp.created_at_history AS created_at,
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
            gmx.call_betweenness AS call_betweenness_ext,
            gmx.call_closeness AS call_closeness_ext,
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
            tgf.proj_degree AS tests_co_tested_degree,
            tgf.proj_weight AS tests_co_tested_weight,
            tgf.proj_clustering AS tests_co_tested_clustering,
            tgf.proj_betweenness AS tests_co_tested_betweenness,
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
            mp.import_fan_in AS module_import_fan_in,
            mp.import_fan_out AS module_import_fan_out,
            mp.in_cycle AS module_in_import_cycle,
            mp.cycle_group AS module_import_cycle_group,
            sm.subsystem_id,
            ss.name AS subsystem_name,
            ss.risk_level AS subsystem_risk_level,
            ss.module_count AS subsystem_module_count
        FROM analytics.function_profile fp
        LEFT JOIN analytics.graph_metrics_functions gm
          ON gm.function_goid_h128 = fp.function_goid_h128
         AND gm.repo = fp.repo
         AND gm.commit = fp.commit
        LEFT JOIN analytics.graph_metrics_functions_ext gmx
          ON gmx.function_goid_h128 = fp.function_goid_h128
         AND gmx.repo = fp.repo
         AND gmx.commit = fp.commit
        LEFT JOIN analytics.test_graph_metrics_functions tgf
          ON tgf.function_goid_h128 = fp.function_goid_h128
         AND tgf.repo = fp.repo
         AND tgf.commit = fp.commit
        LEFT JOIN analytics.cfg_function_metrics cfg_fn
          ON cfg_fn.function_goid_h128 = fp.function_goid_h128
         AND cfg_fn.repo = fp.repo
         AND cfg_fn.commit = fp.commit
        LEFT JOIN analytics.dfg_function_metrics dfg_fn
          ON dfg_fn.function_goid_h128 = fp.function_goid_h128
         AND dfg_fn.repo = fp.repo
         AND dfg_fn.commit = fp.commit
        LEFT JOIN analytics.module_profile mp
          ON mp.module = fp.module
         AND mp.repo = fp.repo
         AND mp.commit = fp.commit
        LEFT JOIN analytics.subsystem_modules sm
          ON sm.module = mp.module
         AND sm.repo = mp.repo
         AND sm.commit = mp.commit
        LEFT JOIN analytics.subsystems ss
          ON ss.subsystem_id = sm.subsystem_id
         AND ss.repo = sm.repo
         AND ss.commit = sm.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_history AS
        SELECT
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
            fh.stability_bucket
        FROM analytics.function_profile fp
        LEFT JOIN analytics.function_history fh
          ON fh.repo = fp.repo
         AND fh.commit = fp.commit
         AND fh.function_goid_h128 = fp.function_goid_h128;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_function_history_timeseries AS
        SELECT
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
            h.bucket_label
        FROM analytics.history_timeseries h
        WHERE h.entity_kind = 'function';
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_module_history_timeseries AS
        SELECT
            h.repo,
            h.entity_stable_id,
            h.commit,
            h.commit_ts,
            h.module,
            h.rel_path,
            h.coverage_ratio,
            h.risk_score,
            h.risk_level,
            h.bucket_label
        FROM analytics.history_timeseries h
        WHERE h.entity_kind = 'module';
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
            mp.role,
            mp.role_confidence,
            mp.role_sources_json,
            gmx.import_betweenness AS import_betweenness_ext,
            gmx.import_closeness AS import_closeness_ext,
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
            cfgm.community_id AS config_community_id,
            cfgm.degree AS config_degree,
            cfgm.weighted_degree AS config_weighted_degree,
            cfgm.betweenness AS config_betweenness,
            cfgm.closeness AS config_closeness,
            sg.import_in_degree AS subsystem_import_in_degree,
            sg.import_out_degree AS subsystem_import_out_degree,
            sg.import_pagerank AS subsystem_import_pagerank,
            sg.import_betweenness AS subsystem_import_betweenness,
            sg.import_closeness AS subsystem_import_closeness,
            sg.import_layer AS subsystem_import_layer,
            sa.import_community_id AS subsystem_agreed_import_community_id,
            sa.agrees AS subsystem_import_agreement,
            gm.created_at
        FROM analytics.graph_metrics_modules gm
        LEFT JOIN core.modules m
          ON m.module = gm.module
        LEFT JOIN analytics.module_profile mp
          ON mp.module = gm.module
         AND mp.repo = gm.repo
         AND mp.commit = gm.commit
        LEFT JOIN analytics.graph_metrics_modules_ext gmx
          ON gmx.module = gm.module
         AND gmx.repo = gm.repo
         AND gmx.commit = gm.commit
        LEFT JOIN analytics.symbol_graph_metrics_modules sym
          ON sym.module = gm.module
         AND sym.repo = gm.repo
         AND sym.commit = gm.commit
        LEFT JOIN analytics.config_graph_metrics_modules cfgm
          ON cfgm.module = gm.module
         AND cfgm.repo = gm.repo
         AND cfgm.commit = gm.commit
        LEFT JOIN analytics.subsystem_modules sm
          ON sm.module = gm.module
         AND sm.repo = gm.repo
         AND sm.commit = gm.commit
        LEFT JOIN analytics.subsystem_graph_metrics sg
          ON sg.subsystem_id = sm.subsystem_id
         AND sg.repo = sm.repo
         AND sg.commit = sm.commit
        LEFT JOIN analytics.subsystem_agreement sa
          ON sa.module = gm.module
         AND sa.repo = gm.repo
         AND sa.commit = gm.commit;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_symbol_module_graph AS
        SELECT
            repo,
            commit,
            module,
            symbol_betweenness,
            symbol_closeness,
            symbol_eigenvector,
            symbol_harmonic,
            symbol_k_core,
            symbol_constraint,
            symbol_effective_size,
            symbol_community_id,
            symbol_component_id,
            symbol_component_size,
            created_at
        FROM analytics.symbol_graph_metrics_modules;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_graph_metrics_keys AS
        SELECT * FROM analytics.config_graph_metrics_keys;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_graph_metrics_modules AS
        SELECT * FROM analytics.config_graph_metrics_modules;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_projection_key_edges AS
        SELECT * FROM analytics.config_projection_key_edges;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_projection_module_edges AS
        SELECT * FROM analytics.config_projection_module_edges;
        """
    )

    con.execute(
        """
CREATE OR REPLACE VIEW docs.v_cfg_block_architecture AS
        SELECT
            fp.function_goid_h128,
            fp.urn AS function_urn,
            fp.repo,
            fp.commit,
            fp.rel_path,
            fp.module,
            fp.kind AS function_kind,
            fp.qualname AS function_qualname,
            fp.risk_level AS function_risk_level,
            fp.risk_score AS function_risk_score,
            cb.block_idx,
            cb.block_id,
            cb.label AS block_label,
            cb.kind AS block_kind,
            cb.file_path AS block_file_path,
            cb.start_line AS block_start_line,
            cb.end_line AS block_end_line,
            cb.in_degree AS cfg_in_degree,
            cb.out_degree AS cfg_out_degree,
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
            bm.loop_nesting_depth
        FROM graph.cfg_blocks cb
        JOIN core.goids g
          ON g.goid_h128 = cb.function_goid_h128
        LEFT JOIN analytics.function_profile fp
          ON fp.function_goid_h128 = cb.function_goid_h128
         AND fp.repo = g.repo
         AND fp.commit = g.commit
        LEFT JOIN analytics.cfg_block_metrics bm
          ON bm.function_goid_h128 = cb.function_goid_h128
         AND bm.repo = g.repo
         AND bm.commit = g.commit
         AND bm.block_idx = cb.block_idx;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_dfg_block_architecture AS
        SELECT
            fp.function_goid_h128,
            fp.urn AS function_urn,
            fp.repo,
            fp.commit,
            fp.rel_path,
            fp.module,
            fp.kind AS function_kind,
            fp.qualname AS function_qualname,
            fp.risk_level AS function_risk_level,
            fp.risk_score AS function_risk_score,
            cb.block_idx,
            cb.block_id,
            cb.label AS block_label,
            cb.kind AS block_kind,
            cb.file_path AS block_file_path,
            cb.start_line AS block_start_line,
            cb.end_line AS block_end_line,
            dbm.dfg_in_degree,
            dbm.dfg_out_degree,
            dbm.dfg_phi_in_degree,
            dbm.dfg_phi_out_degree,
            dbm.dfg_bc_betweenness,
            dbm.dfg_bc_closeness,
            dbm.dfg_bc_eigenvector,
            dbm.dfg_in_chain,
            dbm.dfg_in_scc
        FROM graph.cfg_blocks cb
        JOIN core.goids g
          ON g.goid_h128 = cb.function_goid_h128
        LEFT JOIN analytics.function_profile fp
          ON fp.function_goid_h128 = cb.function_goid_h128
         AND fp.repo = g.repo
         AND fp.commit = g.commit
        LEFT JOIN analytics.dfg_block_metrics dbm
          ON dbm.function_goid_h128 = cb.function_goid_h128
         AND dbm.repo = g.repo
         AND dbm.commit = g.commit
         AND dbm.block_idx = cb.block_idx;
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
         AND agree.subsystem_id = s.subsystem_id;
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
        CREATE OR REPLACE VIEW docs.v_subsystem_agreement AS
        SELECT
            repo,
            commit,
            module,
            subsystem_id,
            import_community_id,
            agrees,
            created_at
        FROM analytics.subsystem_agreement;
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
        CREATE OR REPLACE VIEW docs.v_test_architecture AS
        SELECT
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
            p.created_at
        FROM analytics.test_profile p
        LEFT JOIN analytics.behavioral_coverage b
          ON b.repo = p.repo
         AND b.commit = p.commit
         AND b.test_id = p.test_id;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_behavioral_classification_input AS
        SELECT
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
            b.heuristic_version
        FROM analytics.test_profile p
        LEFT JOIN analytics.behavioral_coverage b
          ON b.repo = p.repo
         AND b.commit = p.commit
         AND b.test_id = p.test_id;
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
        CREATE OR REPLACE VIEW docs.v_entrypoints AS
        SELECT
            repo,
            commit,
            entrypoint_id,
            kind,
            framework,
            handler_goid_h128,
            handler_urn,
            handler_rel_path,
            handler_module,
            handler_qualname,
            http_method,
            route_path,
            status_codes,
            auth_required,
            command_name,
            arguments_schema,
            schedule,
            trigger,
            extra,
            subsystem_id,
            subsystem_name,
            tags,
            owners,
            tests_touching,
            failing_tests,
            slow_tests,
            flaky_tests,
            entrypoint_coverage_ratio,
            last_test_status,
            created_at
        FROM analytics.entrypoints;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_external_dependencies AS
        SELECT
            repo,
            commit,
            dep_id,
            library,
            service_name,
            category,
            function_count,
            callsite_count,
            modules_json,
            usage_modes,
            config_keys,
            risk_level,
            created_at
        FROM analytics.external_dependencies;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_external_dependency_calls AS
        SELECT
            repo,
            commit,
            dep_id,
            library,
            service_name,
            function_goid_h128,
            function_urn,
            rel_path,
            module,
            qualname,
            callsite_count,
            modes,
            evidence_json,
            created_at
        FROM analytics.external_dependency_calls;
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

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_models AS
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
            ) AS fields_json,
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
            ) AS relationships_json,
            dm.doc_short,
            dm.doc_long,
            dm.created_at
        FROM analytics.data_models dm;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_model_fields AS
        SELECT
            repo,
            commit,
            model_id,
            field_name,
            field_type,
            required,
            has_default,
            default_expr,
            constraints_json,
            source,
            rel_path,
            lineno,
            created_at
        FROM analytics.data_model_fields;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_model_relationships AS
        SELECT
            repo,
            commit,
            source_model_id,
            target_model_id,
            target_module,
            target_model_name,
            field_name,
            relationship_kind,
            multiplicity,
            via,
            evidence_json,
            rel_path,
            lineno,
            created_at
        FROM analytics.data_model_relationships;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_models_normalized AS
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
        FROM analytics.data_models dm;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_data_model_usage AS
        SELECT
            u.repo,
            u.commit,
            u.model_id,
            dm.model_name,
            dm.model_kind,
            u.function_goid_h128,
            fp.qualname        AS function_qualname,
            fp.rel_path        AS function_rel_path,
            fp.risk_score,
            fp.coverage_ratio,
            u.usage_kinds_json,
            u.context_json,
            u.evidence_json,
            u.created_at
        FROM analytics.data_model_usage u
        LEFT JOIN analytics.data_models dm
          ON dm.repo = u.repo
         AND dm.commit = u.commit
         AND dm.model_id = u.model_id
        LEFT JOIN analytics.function_profile fp
          ON fp.repo = u.repo
         AND fp.commit = u.commit
         AND fp.function_goid_h128 = u.function_goid_h128;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_config_data_flow AS
        SELECT
            c.repo,
            c.commit,
            c.config_key,
            c.config_path,
            c.function_goid_h128,
            fp.qualname        AS function_qualname,
            fp.rel_path        AS function_rel_path,
            fp.risk_score,
            fp.coverage_ratio,
            c.usage_kind,
            c.evidence_json,
            c.call_chain_id,
            c.call_chain_json,
            c.created_at
        FROM analytics.config_data_flow c
        LEFT JOIN analytics.function_profile fp
          ON fp.repo = c.repo
         AND fp.commit = c.commit
         AND fp.function_goid_h128 = c.function_goid_h128;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_validation_summary AS
        SELECT
            'function' AS domain,
            repo,
            commit,
            CAST(function_goid_h128 AS VARCHAR) AS entity_id,
            issue,
            detail
        FROM analytics.function_validation
        UNION ALL
        SELECT
            'graph' AS domain,
            repo,
            commit,
            CAST(entity_id AS VARCHAR) AS entity_id,
            issue,
            detail
        FROM analytics.graph_validation;
        """
    )
