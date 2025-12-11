"""Tests for TestsIngestPlugin behavior and fallbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.ingestion.plugins.tests_plugin import (
    TestsIngestPlugin,
    get_module_paths,
    resolve_report_file,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.factories.row_factories import sample_pytest_summary, sample_pytest_tests
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_target_context_for_plugin,
    make_resource_case_params,
    write_pytest_report,
)
from tests.ingestion.plugins._wiring import ModulePathCase, run_module_path_resolution_scenarios

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway

EXPECTED_TEST_ROWS = 4
TRUNCATED_LONGREPR_LENGTH = 1000
SAMPLE_TESTS: list[dict[str, object]] = sample_pytest_tests()
EXPECTED_UNICODE_NODEID = SAMPLE_TESTS[1]["nodeid"]
SAMPLE_SUMMARY = sample_pytest_summary()


RESOURCE_CASES = make_resource_case_params()


@pytest.mark.parametrize(
    "options",
    [params for _, params in RESOURCE_CASES],
    ids=[name for name, _ in RESOURCE_CASES],
)
def test_module_path_resolution_scenarios(
    tmp_path: Path, options: dict[str, bool], ingestion_gateway: StorageGateway
) -> None:
    """Shared module path resolution coverage for TestsIngestPlugin."""
    case = ModulePathCase(
        resources_path="pkg/mod.py",
        simulate_resources=options["simulate_resources"],
        simulate_db_fallback=options["simulate_db_fallback"],
        simulate_gateway_failure=options["simulate_gateway_failure"],
    )
    run_module_path_resolution_scenarios(
        lambda _capture: TestsIngestPlugin(),
        get_module_paths,
        tmp_path,
        case=case,
        gateway=ingestion_gateway,
    )


def test_resolve_report_file_prefers_build_dir(tmp_path: Path) -> None:
    """Resolution favors build/test-results/pytest-report.json first."""
    plugin = TestsIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)
    write_pytest_report(ctx.repo_root, filename="pytest-report.json")
    build_report = write_pytest_report(ctx.build_dir, filename="pytest-report.json")

    resolved = resolve_report_file(ctx)

    expect_equal(resolved, build_report)


@pytest.mark.anyio
async def test_execute_ingests_test_results_and_summary(tmp_path: Path) -> None:
    """Happy path: test results and summary rows are written."""
    plugin = TestsIngestPlugin()
    overrides = TargetResourceOverrides(modules=("pkg/mod.py",))
    ctx = build_target_context_for_plugin(
        plugin, tmp_path, config=TargetContextConfig(resources=overrides)
    )
    report = write_pytest_report(
        ctx.build_dir,
        tests=SAMPLE_TESTS,
        summary=SAMPLE_SUMMARY,
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts.get("core.test_results"), EXPECTED_TEST_ROWS)
    rows = ctx.gateway.con.execute(
        "SELECT longrepr FROM core.test_results WHERE repo = ? AND commit = ? ORDER BY nodeid",
        [ctx.repo, ctx.commit],
    ).fetchall()
    longreprs = [row[0] for row in rows if row[0] is not None]
    expect_equal(len(longreprs), 1)
    expect_equal(len(longreprs[0]), TRUNCATED_LONGREPR_LENGTH)
    summary_row = ctx.gateway.con.execute(
        "SELECT passed, failed, skipped, error "
        "FROM core.test_summary WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    expect_equal(summary_row, (1, 1, 1, 1))
    expect_true(report.exists())


@pytest.mark.anyio
async def test_execute_handles_missing_report(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No report should yield success with empty row_counts."""
    plugin = TestsIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)
    caplog.set_level("INFO")

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    assert_logged(caplog.records, level="INFO", containing="No pytest report found")


@pytest.mark.anyio
async def test_execute_fails_on_malformed_report(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed JSON should produce a failed result."""
    plugin = TestsIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)
    bad_path = ctx.build_dir / "test-results" / "pytest-report.json"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{", encoding="utf-8")
    caplog.set_level("WARNING")

    result = await plugin.execute(ctx)

    expect_true(result.success is False)
    expect_true("Failed to read test report" in (result.error_message or ""))
    assert_logged(caplog.records, level="WARNING", containing="Tests ingest failed")
