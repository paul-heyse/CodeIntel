"""Smoke test to ensure docs export validation flag is honored.

These tests use xdist_group to run in the same worker due to cyclopts/pydantic
type adapter caching issues that cause ValidationError when tests run in parallel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.cli.errors import CLI_EXIT_USAGE
from tests._helpers import GatewayOptions, provision_gateway_with_repo
from tests._helpers.builders import FunctionTypesRow, GoidRow, insert_rows
from tests._helpers.cli import run_cli

if TYPE_CHECKING:
    from pathlib import Path


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
    insert_rows(
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
    # Insert data with wrong type for function_goid_h128 (string instead of int)
    # This should trigger validation failure since the schema expects integer
    con.execute(
        """
        INSERT INTO analytics.function_profile (
            function_goid_h128,
            urn,
            repo,
            commit,
            rel_path
        ) VALUES
            (1, 'urn:demo', 'demo/repo', 'deadbeef', 'src/file.py')
        """
    )
    # Note: The validation now uses generated JSON Schemas which are more permissive
    # about NULL values. To test validation failure, we rely on the contract validation
    # catching inconsistencies in the data (e.g., missing required fields in related tables)
    insert_rows(
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


@pytest.mark.xdist_group("cli_shared_flags")
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
        "--skip-prereqs",
        "--validate",
    ]
    result = run_cli(args_validate)
    if result.exit_code != 1:
        pytest.fail(f"Expected validation failure exit code 1, got {result.exit_code}")
    if "Validation failed" not in result.stderr:
        pytest.fail("Expected validation failure message in stderr")

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
        "--skip-prereqs",
    ]
    result_no_validate = run_cli(args_no_validate)
    if result_no_validate.exit_code != 0:
        pytest.fail(
            f"Expected success exit code 0 without validation, got {result_no_validate.exit_code}"
        )


@pytest.mark.xdist_group("cli_shared_flags")
def test_docs_export_usage_error_exit_code(tmp_path: Path) -> None:
    """Unknown flags should produce a usage error exit code 2."""
    result = run_cli(
        [
            "docs",
            "export",
            "--unknown-flag",
            "value",
            "--repo-root",
            str(tmp_path),
        ],
    )
    if result.exit_code != CLI_EXIT_USAGE:
        pytest.fail(f"Expected usage error exit code 2, got {result.exit_code}")
    if "No such option" not in result.stderr:
        pytest.fail("Expected usage error message in stderr")
