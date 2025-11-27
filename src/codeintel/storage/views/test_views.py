"""Docs views for test analytics."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

TEST_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_test_to_function",
    "docs.v_test_architecture",
    "docs.v_behavioral_classification_input",
)


def create_test_views(con: DuckDBPyConnection) -> None:
    """Create or replace test-centric docs views."""
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
