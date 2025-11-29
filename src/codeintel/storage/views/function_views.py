"""Function-focused docs views."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

FUNCTION_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_function_summary",
    "docs.v_function_architecture",
    "docs.v_function_history",
    "docs.v_function_history_timeseries",
    "docs.v_cfg_block_architecture",
    "docs.v_dfg_block_architecture",
)


def create_function_views(con: DuckDBPyConnection) -> None:
    """Create or replace docs views for function summaries and profiles."""
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
