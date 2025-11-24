"""Tests for validate_exports tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.docs_export.validate_exports import main
from codeintel.services.errors import ExportError
from tests._helpers.gateway import open_fresh_duckdb, seed_tables


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_validate_jsonl_happy_path(tmp_path: Path) -> None:
    """Validator should return 0 for conforming JSONL rows."""
    data_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": 1,
                "callee_goid_h128": 2,
                "callsite_path": "a.py",
                "language": "python",
            }
        ],
    )
    exit_code = main(["--schema", "call_graph_edges", str(data_path)])
    if exit_code != 0:
        message = f"Expected success, got exit code {exit_code}"
        pytest.fail(message)


def test_validate_jsonl_failure(tmp_path: Path) -> None:
    """Validator should fail when required fields are missing."""
    data_path = tmp_path / "edges.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "commit": "c",
                "caller_goid_h128": 1,
                "callee_goid_h128": 2,
            }
        ],
    )
    exit_code = main(["--schema", "call_graph_edges", str(data_path)])
    if exit_code == 0:
        message = "Expected validation failure for missing repo field"
        pytest.fail(message)


def test_export_raises_on_validation_failure(tmp_path: Path) -> None:
    """Export functions should raise ExportError when schema validation fails."""
    db_path = tmp_path / "db.duckdb"
    gateway = open_fresh_duckdb(db_path)
    con = gateway.con
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    con.execute(
        """
        CREATE TABLE analytics.function_profile (
            function_goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT,
            rel_path TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO analytics.function_profile VALUES (1, 'urn:foo', NULL, 'c', 'foo.py');
        """
    )
    con.execute(
        """
        CREATE TABLE core.repo_map (
            repo TEXT,
            commit TEXT,
            modules JSON,
            overlays JSON,
            generated_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO core.repo_map VALUES ('r','c','{}','{}', CURRENT_TIMESTAMP);
        """
    )
    gateway.close()

    output_dir = tmp_path / "out"
    gw_parquet = open_fresh_duckdb(db_path)
    with pytest.raises(ExportError):
        export_all_parquet(
            gw_parquet,
            output_dir,
            validate_exports=True,
            schemas=["function_profile"],
        )
    gw_parquet.close()


def test_export_logs_problem_detail_on_validation_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Validation failures should log ProblemDetails and raise ExportError."""
    db_path = tmp_path / "db.duckdb"
    gateway = open_fresh_duckdb(db_path)
    seed_tables(
        gateway,
        [
            "CREATE SCHEMA IF NOT EXISTS core;",
            "CREATE SCHEMA IF NOT EXISTS analytics;",
            "DROP TABLE IF EXISTS analytics.function_profile;",
            """
            CREATE TABLE analytics.function_profile (
                function_goid_h128 DECIMAL(38,0),
                urn TEXT,
                repo TEXT,
                commit TEXT,
                rel_path TEXT
            );
            """,
            "DROP TABLE IF EXISTS core.repo_map;",
            """
            CREATE TABLE core.repo_map (
                repo TEXT,
                commit TEXT,
                modules JSON,
                overlays JSON,
                generated_at TIMESTAMP
            );
            """,
        ],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.function_profile VALUES (1, 'urn:foo', NULL, 'c', 'foo.py');
        """
    )
    gateway.con.execute(
        """
        INSERT INTO core.repo_map VALUES ('r','c','{}','{}', CURRENT_TIMESTAMP);
        """
    )
    gateway.close()

    output_dir = tmp_path / "out_jsonl"
    caplog.set_level("ERROR")
    gw_jsonl = open_fresh_duckdb(db_path)
    with pytest.raises(ExportError):
        export_all_jsonl(
            gw_jsonl,
            output_dir,
            validate_exports=True,
            schemas=["function_profile"],
        )
    gw_jsonl.close()
    error_logs = [rec for rec in caplog.records if "export.validation_failed" in rec.getMessage()]
    if not error_logs:
        pytest.fail("Expected ProblemDetail log entry for validation failure")
