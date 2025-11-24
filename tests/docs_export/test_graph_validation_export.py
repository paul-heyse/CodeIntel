"""Ensure graph validation findings are exported to JSONL/Parquet."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.storage.gateway import StorageConfig, open_gateway


def test_graph_validation_export(tmp_path: Path) -> None:
    """
    Graph validation rows should be exported to JSONL and Parquet.

    Raises
    ------
    AssertionError
        If expected export artifacts or content are missing.
    """
    db_path = tmp_path / "db.duckdb"
    gateway = open_gateway(
        StorageConfig(db_path=db_path, apply_schema=True, ensure_views=False, validate_schema=True)
    )
    con = gateway.con
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.graph_validation (
            repo TEXT,
            commit TEXT,
            issue TEXT,
            severity TEXT,
            rel_path TEXT,
            detail TEXT,
            metadata JSON,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO analytics.graph_validation
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "demo/repo",
            "deadbeef",
            "missing_function_goids",
            "warning",
            "pkg/a.py",
            "1 functions, 0 GOIDs",
            {"function_count": 1, "goid_count": 0},
            datetime.now(UTC),
        ],
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
