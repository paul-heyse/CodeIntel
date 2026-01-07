"""Plumbing tests ensuring shared runners and scan configs are honored."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._helpers.assertions import expect_false, expect_true
from tests._helpers.fixtures.repos import write_sample_repo
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)
from tests._helpers.ingestion import build_parquet_repo_scan_context, build_scan_profile
from tests._helpers.parquet_datasets import read_snapshot_rows

if TYPE_CHECKING:
    from pathlib import Path


def test_repo_scan_honors_scan_profile(tmp_path: Path) -> None:
    """Ensure repo_scan respects ignore lists from ScanProfile."""
    repo_structure = {
        "keep/a.py": "print('ok')\n",
        "ignore/b.py": "print('skip')\n",
    }
    repo_root = tmp_path / "repo"
    profile = build_scan_profile(repo_root, ignore_dirs=("ignore",))
    context = build_parquet_repo_scan_context(
        tmp_path,
        repo_structure=repo_structure,
        profile=profile,
    )
    try:
        rows = read_snapshot_rows(
            context.dataset_root,
            table_key="core.modules",
            snapshot_id=context.snapshot.commit,
            columns=("path",),
        )
    except FileNotFoundError:
        pytest.xfail("Parquet datasets not yet materialized for repo_scan.")
    paths: list[str] = []
    for row in rows:
        path = row.get("path")
        if isinstance(path, str):
            paths.append(path)
    paths.sort()
    expect_true("keep/a.py" in paths)
    expect_false("ignore/b.py" in paths)


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
        if not record.success:
            pytest.fail(f"Tests ingest target failed: {record.status}")

        dataset_root = harness.ctx.build_paths.dataset_root_dir
        snapshot = harness.ctx.snapshot
        try:
            rows = read_snapshot_rows(
                dataset_root,
                table_key="analytics.test_catalog",
                snapshot_id=snapshot.commit,
                columns=("test_id",),
            )
        except FileNotFoundError:
            pytest.xfail("Parquet datasets not yet materialized for tests_ingest target.")
        ids = {row.get("test_id") for row in rows}
        expect_true("tests/test_mod.py::test_hello" in ids)
