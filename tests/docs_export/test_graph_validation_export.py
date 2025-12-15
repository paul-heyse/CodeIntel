"""Ensure graph validation findings are exported to JSONL/Parquet."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.exports import export_all_jsonl, export_all_parquet
from codeintel.config.datasets.columns import serialize_row
from tests._helpers import provision_docs_export_ready

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsGraphValidationRow as GraphValidationRow,
    )


GRAPH_VALIDATION_COLUMNS: tuple[str, ...] = (
    "repo",
    "commit",
    "graph_name",
    "entity_id",
    "issue",
    "severity",
    "rel_path",
    "detail",
    "metadata",
    "created_at",
)


@pytest.mark.usefixtures("fresh_gateway")
def test_graph_validation_export(tmp_path: Path) -> None:
    """
    Graph validation rows should be exported to JSONL and Parquet.

    Raises
    ------
    AssertionError
        If expected export artifacts or content are missing.
    """
    ctx = provision_docs_export_ready(
        tmp_path, repo="demo/repo", commit="deadbeef", file_backed=False
    )
    gateway = ctx.gateway
    con = gateway.con
    con.execute(
        "DELETE FROM analytics.graph_validation WHERE repo = ? AND commit = ?",
        ["demo/repo", "deadbeef"],
    )
    row: GraphValidationRow = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "graph_name": "call_graph",
        "entity_id": "pkg/a.py",
        "issue": "missing_function_goids",
        "severity": "warning",
        "rel_path": "pkg/a.py",
        "detail": "1 functions, 0 GOIDs",
        "metadata": {"function_count": 1, "goid_count": 0},
        "created_at": datetime.now(UTC),
    }
    con.execute(
        """
        INSERT INTO analytics.graph_validation (
            repo, commit, graph_name, entity_id, issue, severity, rel_path, detail, metadata, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        serialize_row(row, GRAPH_VALIDATION_COLUMNS),
    )

    doc_out = tmp_path / "Document Output"
    export_all_jsonl(gateway, doc_out)
    export_all_parquet(gateway, doc_out)

    jsonl_path = doc_out / "graph_validation.jsonl"
    parquet_path = doc_out / "graph_validation.parquet"
    if not jsonl_path.is_file():
        message = "graph_validation.jsonl not written"
        raise AssertionError(message)
    if not parquet_path.is_file():
        message = "graph_validation.parquet not written"
        raise AssertionError(message)

    json_content = jsonl_path.read_text(encoding="utf8").strip()
    if "missing_function_goids" not in json_content:
        message = "graph_validation JSONL missing expected check name"
        raise AssertionError(message)

    count = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(parquet_path)]).fetchone()
    if count is None or int(count[0]) != 1:
        message = f"graph_validation Parquet row count unexpected: {count}"
        raise AssertionError(message)
