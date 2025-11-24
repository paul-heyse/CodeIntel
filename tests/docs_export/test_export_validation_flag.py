"""Smoke test to ensure docs export validation flag is honored."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from codeintel.cli import main as cli_main


def _seed_invalid_function_profile(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
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
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES ('demo/repo', 'deadbeef', '{}', '{}', CURRENT_TIMESTAMP);
        """
    )
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
        INSERT INTO analytics.function_profile (
            function_goid_h128,
            urn,
            repo,
            commit,
            rel_path
        ) VALUES
            (1, 'urn:demo', 'demo/repo', NULL, 'src/file.py');
        """
    )
    con.close()


def test_docs_export_validation_flag_triggers_schema_check(tmp_path: Path) -> None:
    """Verify docs export honors validation toggle and surfaces failures."""
    db_path = tmp_path / "db.duckdb"
    _seed_invalid_function_profile(db_path)

    output_dir = tmp_path / "out_validate"
    args_validate = [
        "docs",
        "export",
        "--repo-root",
        str(tmp_path),
        "--repo",
        "demo/repo",
        "--commit",
        "deadbeef",
        "--db-path",
        str(db_path),
        "--build-dir",
        str(tmp_path / "build"),
        "--document-output-dir",
        str(output_dir),
        "--validate",
    ]
    exit_code = cli_main.main(args_validate)
    if exit_code != 1:
        pytest.fail(f"Expected validation failure exit code 1, got {exit_code}")

    output_dir_no_validate = tmp_path / "out_no_validate"
    args_no_validate = [
        "docs",
        "export",
        "--repo-root",
        str(tmp_path),
        "--repo",
        "demo/repo",
        "--commit",
        "deadbeef",
        "--db-path",
        str(db_path),
        "--build-dir",
        str(tmp_path / "build2"),
        "--document-output-dir",
        str(output_dir_no_validate),
    ]
    exit_code_no_validate = cli_main.main(args_no_validate)
    if exit_code_no_validate != 0:
        pytest.fail(f"Expected success exit code 0 without validation, got {exit_code_no_validate}")
