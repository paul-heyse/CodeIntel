"""IDE-facing docs views."""

from __future__ import annotations

from duckdb import DuckDBPyConnection

IDE_VIEW_NAMES: tuple[str, ...] = ("docs.v_ide_hints",)


def create_ide_views(con: DuckDBPyConnection) -> None:
    """Create or replace IDE-facing docs views."""
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
