"""Smoke test to ensure docs export validation flag is honored."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.cli import main as cli_main
from tests._helpers.fixtures import GatewayOptions, provision_gateway_with_repo


def _seed_invalid_function_profile(db_path: Path, repo_root: Path) -> None:
    ctx = provision_gateway_with_repo(
        repo_root,
        repo="demo/repo",
        commit="deadbeef",
        options=GatewayOptions(
            db_path=db_path,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            file_backed=True,
        ),
    )
    con = ctx.gateway.con
    con.execute("DELETE FROM analytics.function_profile")
    con.execute("DELETE FROM core.repo_map")
    con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES ('demo/repo', 'deadbeef', '{}', '{}', CURRENT_TIMESTAMP)
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
            (1, 'urn:demo', 'demo/repo', NULL, 'src/file.py')
        """
    )
    ctx.close()


def test_docs_export_validation_flag_triggers_schema_check(tmp_path: Path) -> None:
    """Verify docs export honors validation toggle and surfaces failures."""
    db_path = tmp_path / "db.duckdb"
    _seed_invalid_function_profile(db_path, tmp_path)

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
