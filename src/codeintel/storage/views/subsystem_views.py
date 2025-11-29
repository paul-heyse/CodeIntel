"""Subsystem docs views."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

SUBSYSTEM_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_subsystem_summary",
    "docs.v_module_with_subsystem",
    "docs.v_subsystem_agreement",
    "docs.v_subsystem_profile",
    "docs.v_subsystem_coverage",
)


def create_subsystem_views(con: DuckDBPyConnection) -> None:
    """Create or replace subsystem docs views."""
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
        CREATE OR REPLACE VIEW docs.v_subsystem_profile AS
        SELECT
            s.repo,
            s.commit,
            s.subsystem_id,
            coalesce(c.name, s.name) AS name,
            coalesce(c.description, s.description) AS description,
            coalesce(c.module_count, s.module_count) AS module_count,
            coalesce(c.modules_json, s.modules_json) AS modules_json,
            coalesce(c.entrypoints_json, s.entrypoints_json, '[]') AS entrypoints_json,
            coalesce(c.internal_edge_count, s.internal_edge_count, 0) AS internal_edge_count,
            coalesce(c.external_edge_count, s.external_edge_count, 0) AS external_edge_count,
            coalesce(c.fan_in, s.fan_in, 0) AS fan_in,
            coalesce(c.fan_out, s.fan_out, 0) AS fan_out,
            coalesce(c.function_count, s.function_count, 0) AS function_count,
            coalesce(c.avg_risk_score, s.avg_risk_score) AS avg_risk_score,
            coalesce(c.max_risk_score, s.max_risk_score) AS max_risk_score,
            coalesce(
                c.high_risk_function_count,
                s.high_risk_function_count,
                0
            ) AS high_risk_function_count,
            coalesce(c.risk_level, s.risk_level) AS risk_level,
            coalesce(c.import_in_degree, gm.import_in_degree) AS import_in_degree,
            coalesce(c.import_out_degree, gm.import_out_degree) AS import_out_degree,
            coalesce(c.import_pagerank, gm.import_pagerank) AS import_pagerank,
            coalesce(c.import_betweenness, gm.import_betweenness) AS import_betweenness,
            coalesce(c.import_closeness, gm.import_closeness) AS import_closeness,
            coalesce(c.import_layer, gm.import_layer, 0) AS import_layer,
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
          AND s.commit IS NOT NULL;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW docs.v_subsystem_coverage AS
        WITH coverage AS (
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
        )
        SELECT
            s.repo,
            s.commit,
            s.subsystem_id,
            coalesce(cc.name, s.name) AS name,
            coalesce(cc.description, s.description) AS description,
            coalesce(cc.module_count, s.module_count) AS module_count,
            coalesce(cc.function_count, s.function_count) AS function_count,
            coalesce(cc.risk_level, s.risk_level) AS risk_level,
            coalesce(cc.avg_risk_score, s.avg_risk_score) AS avg_risk_score,
            coalesce(cc.max_risk_score, s.max_risk_score) AS max_risk_score,
            coalesce(cc.test_count, c.test_count, 0) AS test_count,
            coalesce(cc.passed_test_count, c.passed_test_count, 0) AS passed_test_count,
            coalesce(cc.failed_test_count, c.failed_test_count, 0) AS failed_test_count,
            coalesce(cc.skipped_test_count, c.skipped_test_count, 0) AS skipped_test_count,
            coalesce(cc.xfail_test_count, c.xfail_test_count, 0) AS xfail_test_count,
            coalesce(cc.flaky_test_count, c.flaky_test_count, 0) AS flaky_test_count,
            coalesce(
                cc.total_functions_covered,
                c.total_functions_covered,
                0
            ) AS total_functions_covered,
            coalesce(cc.avg_functions_covered, c.avg_functions_covered, 0) AS avg_functions_covered,
            coalesce(cc.max_functions_covered, c.max_functions_covered, 0) AS max_functions_covered,
            coalesce(cc.min_functions_covered, c.min_functions_covered, 0) AS min_functions_covered,
            CASE
                WHEN coalesce(cc.function_count, s.function_count, 0) = 0 THEN NULL
                ELSE coalesce(cc.total_functions_covered, c.total_functions_covered, 0) * 1.0
                     / coalesce(cc.function_count, s.function_count)
            END AS function_coverage_ratio,
            coalesce(cc.created_at, s.created_at) AS created_at
        FROM analytics.subsystems s
        LEFT JOIN coverage c
          ON c.repo = s.repo
         AND c.commit = s.commit
         AND c.subsystem_id = s.subsystem_id
        LEFT JOIN analytics.subsystem_coverage_cache cc
          ON cc.repo = s.repo
         AND cc.commit = s.commit
         AND cc.subsystem_id = s.subsystem_id
        WHERE s.repo IS NOT NULL
          AND s.commit IS NOT NULL;
        """
    )
