"""Metadata view definitions for the meta catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.storage.helpers.table_key import fully_qualified_table_ref

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = ["apply_metadata_views"]


def apply_metadata_views(con: DuckDBPyConnection, *, catalog: str | None) -> None:
    """Create or replace metadata views."""
    summary_view = fully_qualified_table_ref(
        "metadata.v_schema_validation_summary",
        catalog=catalog,
    )
    failures_view = fully_qualified_table_ref(
        "metadata.v_schema_validation_failures",
        catalog=catalog,
    )
    latest_good_view = fully_qualified_table_ref(
        "metadata.v_schema_manifest_latest_good",
        catalog=catalog,
    )
    runs_ref = fully_qualified_table_ref(
        "metadata.schema_validation_runs",
        catalog=catalog,
    )
    manifest_runs_ref = fully_qualified_table_ref(
        "metadata.schema_manifest_runs",
        catalog=catalog,
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {summary_view} AS
        SELECT
            validation_id,
            repo,
            commit,
            validation_mode,
            include_views,
            status,
            issue_count,
            created_at
        FROM (
            SELECT
                validation_id,
                repo,
                commit,
                validation_mode,
                include_views,
                status,
                issue_count,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY repo, commit
                    ORDER BY created_at DESC
                ) AS row_num
            FROM {runs_ref}
        ) AS ranked
        WHERE row_num = 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {failures_view} AS
        SELECT
            validation_id,
            repo,
            commit,
            validation_mode,
            include_views,
            status,
            issue_count,
            issues,
            created_at
        FROM (
            SELECT
                validation_id,
                repo,
                commit,
                validation_mode,
                include_views,
                status,
                issue_count,
                issues,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY repo, commit
                    ORDER BY created_at DESC
                ) AS row_num
            FROM {runs_ref}
            WHERE status = 'failed'
        ) AS ranked
        WHERE row_num = 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {latest_good_view} AS
        SELECT
            repo,
            commit,
            manifest_kind,
            catalog_hash,
            created_at AS manifest_created_at,
            validation_status,
            validation_created_at
        FROM (
            SELECT
                m.repo,
                m.commit,
                m.manifest_kind,
                m.catalog_hash,
                m.created_at,
                s.status AS validation_status,
                s.created_at AS validation_created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY m.repo
                    ORDER BY m.created_at DESC
                ) AS row_num
            FROM {manifest_runs_ref} AS m
            JOIN {summary_view} AS s
              ON m.repo = s.repo AND m.commit = s.commit
            WHERE s.status = 'passed' AND s.created_at >= m.created_at
        ) AS ranked
        WHERE row_num = 1
        """
    )
