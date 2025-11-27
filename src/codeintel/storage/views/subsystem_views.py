"""Subsystem docs views."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

SUBSYSTEM_VIEW_NAMES: tuple[str, ...] = (
    "docs.v_subsystem_summary",
    "docs.v_module_with_subsystem",
    "docs.v_subsystem_agreement",
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
