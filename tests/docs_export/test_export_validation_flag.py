"""Smoke test to ensure docs export validation flag is honored."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.cli import main as cli_main
from tests._helpers.builders import FunctionTypesRow, GoidRow, insert_function_types, insert_goids
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
    now = datetime.now(tz=UTC)
    insert_goids(
        ctx.gateway,
        [
            GoidRow(
                goid_h128=1,
                urn="urn:demo",
                repo="demo/repo",
                commit="deadbeef",
                rel_path="src/file.py",
                kind="function",
                qualname="demo.fn",
                start_line=1,
                end_line=2,
            )
        ],
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
    insert_function_types(
        ctx.gateway,
        [
            FunctionTypesRow(
                function_goid_h128=1,
                urn="urn:demo",
                repo="demo/repo",
                commit="deadbeef",
                rel_path="src/file.py",
                language="python",
                kind="function",
                qualname="demo.fn",
                start_line=1,
                end_line=2,
                total_params=0,
                annotated_params=0,
                unannotated_params=0,
                param_typed_ratio=0.0,
                has_return_annotation=False,
                return_type="",
                return_type_source="annotation",
                type_comment=None,
                param_types_json="{}",
                fully_typed=False,
                partial_typed=False,
                untyped=True,
                typedness_bucket="untyped",
                typedness_source="manual",
                created_at=now,
            )
        ],
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
