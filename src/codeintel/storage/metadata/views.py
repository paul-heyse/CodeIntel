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
    runs_ref = fully_qualified_table_ref(
        "metadata.schema_validation_runs",
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
