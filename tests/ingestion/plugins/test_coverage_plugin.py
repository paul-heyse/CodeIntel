"""Tests for CoverageIngestPlugin wiring and fallbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.protocols import CoverageData
from codeintel.build.providers import Providers
from codeintel.build.context import TargetExecutionContext
from codeintel.storage.gateway import StorageGateway
from codeintel.ingestion.plugins.coverage_plugin import (
    CoverageIngestPlugin,
    get_module_paths,
    paths_to_modules,
    resolve_coverage_file,
)
from tests._helpers.assertions import assert_logged, expect_equal, expect_true
from tests._helpers.factories.row_factories import sample_coverage_payload
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.fake_providers import FakeCoverageCollector, FakeProviders
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_repo_tree,
    build_target_context_for_plugin,
    make_resource_case_params,
    run_ingestion_scenario,
    write_coverage_file,
)
from tests.ingestion.plugins._wiring import ModulePathCase, run_module_path_resolution_scenarios


def test_paths_to_modules_builds_metadata(tmp_path: Path) -> None:
    """Ensure path conversion sets module names, file paths, and ordering."""
    repo_root = tmp_path / "repo"
    paths = ["pkg/mod.py", "pkg/util/helpers.py"]
    modules = paths_to_modules(paths, repo_root)

    expect_equal(modules[0].module_name, "pkg.mod")
    expect_equal(modules[0].file_path, repo_root / "pkg/mod.py")
    expect_equal(modules[0].index, 1)
    expect_equal(modules[0].total, len(paths))
    expect_equal(modules[1].module_name, "pkg.util.helpers")


RESOURCE_CASES = make_resource_case_params()


@pytest.mark.parametrize(
    "options",
    [params for _, params in RESOURCE_CASES],
    ids=[name for name, _ in RESOURCE_CASES],
)
def test_module_path_resolution_scenarios(
    tmp_path: Path, options: dict[str, bool], ingestion_gateway: StorageGateway
) -> None:
    """Shared coverage of module path resolution for CoverageIngestPlugin."""
    case = ModulePathCase(
        resources_path="pkg/mod.py",
        simulate_resources=options["simulate_resources"],
        simulate_db_fallback=options["simulate_db_fallback"],
        simulate_gateway_failure=options["simulate_gateway_failure"],
    )
    run_module_path_resolution_scenarios(
        lambda _capture: CoverageIngestPlugin(),
        get_module_paths,
        tmp_path,
        case=case,
        gateway=ingestion_gateway,
    )


@pytest.mark.anyio
async def test_execute_skips_when_no_coverage_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When no coverage file is found, plugin should log and return empty counts."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    caplog.set_level(logging.INFO)

    _ctx, result = await run_ingestion_scenario(
        CoverageIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root),
    )

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    assert_logged(caplog.records, level="INFO", containing="No coverage file found")


def test_resolve_coverage_file_prefers_repo_dot_coverage(tmp_path: Path) -> None:
    """Resolution favors repo-root .coverage over other candidates."""
    plugin = CoverageIngestPlugin()
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root),
    )
    write_coverage_file(ctx.build_dir, filename="coverage.json", content="{}")
    repo_cov = write_coverage_file(repo_root, filename=".coverage", content="binary-ish")

    resolved = resolve_coverage_file(ctx)

    expect_equal(resolved, repo_cov)


@pytest.mark.anyio
async def test_execute_ingests_coverage_with_fake_collector(tmp_path: Path) -> None:
    """Happy path: coverage rows are written using the fake collector."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/mod.py": "x = 1\n", "pkg/naïve.py": "y = 2\n"},
    )
    coverage_payload = sample_coverage_payload()
    coverage_file = repo_root / ".coverage"

    fake_providers = FakeProviders()
    fake_providers.coverage_collector.coverage_data = {
        "pkg/mod.py": CoverageData(
            path="pkg/mod.py",
            covered_lines=frozenset({1, 2}),
            missing_lines=frozenset({3}),
        ),
        "pkg/naïve.py": CoverageData(
            path="pkg/naïve.py",
            covered_lines=frozenset({1}),
            missing_lines=frozenset({2, 3}),
        ),
    }
    overrides = TargetResourceOverrides(
        providers=cast("Providers", fake_providers),
        modules=("pkg/mod.py", "pkg/naïve.py"),
    )

    def _seed_coverage(context: TargetExecutionContext) -> None:
        write_coverage_file(
            coverage_file.parent,
            filename=coverage_file.name,
            content=coverage_payload,
        )

    ctx, result = await run_ingestion_scenario(
        CoverageIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
        seed_fn=_seed_coverage,
    )

    expect_true(result.success is True)
    expected_rows = 6
    expect_equal(result.row_counts.get("analytics.coverage_lines"), expected_rows)
    row = ctx.gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.coverage_lines WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    if row is None:
        pytest.fail("Coverage rows should exist for analytics.coverage_lines")
    expect_equal(row[0], expected_rows)
    calls = fake_providers.coverage_collector.collect_calls.calls
    expect_equal(len(calls), 1)
    expect_equal(calls[0].path, coverage_file)


class _FailingCollector:
    """Fake coverage collector that always fails."""

    @staticmethod
    async def collect(coverage_file: Path) -> object:
        """Raise to simulate tool failure.

        Parameters
        ----------
        coverage_file
            Path to the coverage artifact.

        Raises
        ------
        RuntimeError
            Always raised to simulate a tool failure.
        """
        _ = coverage_file
        message = "boom"
        raise RuntimeError(message)


@pytest.mark.anyio
async def test_execute_fails_when_collector_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing or failing collector should produce a failed result."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    write_coverage_file(repo_root, filename=".coverage", content="{}")
    overrides = TargetResourceOverrides(providers=None, modules=("pkg/mod.py",))
    caplog.set_level(logging.WARNING)

    def _seed_missing(_context: TargetExecutionContext) -> None:
        write_coverage_file(repo_root, filename=".coverage", content="{}")

    _ctx, result = await run_ingestion_scenario(
        CoverageIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
        seed_fn=_seed_missing,
    )

    expect_true(result.success is False)
    expect_true("Coverage ingest failed" in (result.error_message or ""))
    expect_true("Coverage collector not available" in (result.error_message or ""))
    expect_true("Coverage export failed" in caplog.text)


@pytest.mark.anyio
async def test_execute_fails_when_collector_raises(tmp_path: Path) -> None:
    """Collector exceptions propagate as failed plugin results."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})

    failing_providers = FakeProviders()
    failing_providers.coverage_collector = cast("FakeCoverageCollector", _FailingCollector())
    overrides = TargetResourceOverrides(
        providers=cast("Providers", failing_providers),
        modules=("pkg/mod.py",),
    )

    def _seed_raise(_context: TargetExecutionContext) -> None:
        write_coverage_file(repo_root, filename=".coverage", content="{}")

    _ctx, result = await run_ingestion_scenario(
        CoverageIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
        seed_fn=_seed_raise,
    )

    expect_true(result.success is False)
    expect_true("Coverage ingest failed" in (result.error_message or ""))
