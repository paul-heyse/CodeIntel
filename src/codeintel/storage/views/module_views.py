"""Module- and file-level docs views."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

MODULE_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_module_history_timeseries",
    "docs.v_module_architecture",
    "docs.v_file_summary",
    "docs.v_entrypoints",
    "docs.v_external_dependencies",
    "docs.v_external_dependency_calls",
    "docs.v_file_profile",
    "docs.v_module_profile",
)


def create_module_views(con: DuckDBPyConnection) -> None:
    """Create or replace module/file docs views."""
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
