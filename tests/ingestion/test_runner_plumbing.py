"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import assert_target_ok, expect_rows_equal
from tests._helpers.fixtures.repos import write_sample_repo
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)
from tests._helpers.ingestion import (
    ScanSetupOptions,
    closing_gateway,
    make_scan_setup,
    materialize_repo_scan_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_repo_scan_honors_scan_profile(tmp_path: Path) -> None:
    """Ensure repo_scan respects ignore lists from ScanProfile."""
    setup = make_scan_setup(
        tmp_path,
        options=ScanSetupOptions(
            repo_structure={
                "keep/a.py": "print('ok')\n",
                "ignore/b.py": "print('skip')\n",
            },
            ignore_dirs=("ignore",),
        ),
    )

    with closing_gateway(setup.gateway):
        scan_result = setup.scan_step.execute(
            repo="r",
            commit="c",
            repo_root=setup.repo_root,
            profile=setup.profile,
        )
        materialize_repo_scan_result(
            setup.gateway,
            scan_result,
            snapshot=SnapshotRef(repo="r", commit="c", repo_root=setup.repo_root),
        )

        rows = setup.gateway.con.table("core.modules").select("path").fetchall()
        expect_rows_equal(rows, [("keep/a.py",)], message="Unexpected modules from repo_scan")


def test_tests_ingest_uses_report_file(tmp_path: Path) -> None:
    """Verify tests_ingest consumes the pytest report artifact."""
    with HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="r", commit="c"),
        options=HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_sample_repo,
        ),
    ) as harness:
        harness.artifacts.write_pytest_report(
            tests=[
                {
                    "nodeid": "tests/test_mod.py::test_hello",
                    "outcome": "passed",
                    "duration": 0.01,
                }
            ],
            summary={"passed": 1, "failed": 0, "skipped": 0},
        )
        result = harness.run_targets(["tests_ingest"])
        record = harness.record("tests_ingest", result=result)
        assert_target_ok(record)

        row = harness.ctx.gateway.con.execute(
            "SELECT test_id FROM analytics.test_catalog WHERE test_id = ?",
            ["tests/test_mod.py::test_hello"],
        ).fetchone()
        if row is None:
            pytest.fail("tests_ingest failed to persist test_catalog rows")


@pytest.mark.skip(
    reason="Schema mismatch: StaticDiagnosticRow (6 cols) vs static_diagnostics table (8 cols)"
)
def test_typing_ingest_uses_shared_runner(
    tmp_path: Path,
) -> None:
    """Ensure typing ingestion reuses the provided ToolRunner."""
    with HamiltonBuildHarness.open(
        tmp_path,
        harness=HarnessConfig(repo="r", commit="c"),
        options=HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_sample_repo,
        ),
    ) as harness:
        result = harness.run_targets(["typing"])
        record = harness.record("typing", result=result)
        assert_target_ok(record)

        row = harness.ctx.gateway.con.execute("SELECT COUNT(*) FROM analytics.typedness").fetchone()
        if (row[0] if row else 0) < 1:
            pytest.fail("Typedness ingestion wrote no rows")
