"""Tests for CoverageIngestPlugin wiring and fallbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.protocols import CoverageData
from codeintel.build.providers import Providers
from codeintel.ingestion.plugins.coverage_plugin import (
    CoverageIngestPlugin,
    get_module_paths,
    paths_to_modules,
    resolve_coverage_file,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.fake_providers import FakeCoverageCollector, FakeProviders
from tests._helpers.fakes.recording_gateways import FailingGateway
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_target_context_for_plugin,
    write_coverage_file,
)


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


def test_get_module_paths_prefers_resources(tmp_path: Path) -> None:
    """When modules are provided in resources, DB is not consulted."""
    plugin = CoverageIngestPlugin()
    overrides = TargetResourceOverrides(modules=("pkg/mod.py",))
    ctx = build_target_context_for_plugin(
        plugin, tmp_path, config=TargetContextConfig(resources=overrides)
    )
    ctx.gateway.con.execute("DELETE FROM core.modules")

    paths = get_module_paths(ctx)

    expect_equal(paths, ["pkg/mod.py"])


def test_get_module_paths_reads_database_when_missing(tmp_path: Path) -> None:
    """DB rows are used when resources.modules is empty."""
    plugin = CoverageIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)
    ctx.gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["pkg.mod", "pkg/mod.py", DEFAULT_REPO, DEFAULT_COMMIT],
    )

    paths = get_module_paths(ctx)

    expect_equal(paths, ["pkg/mod.py"])


def test_get_module_paths_handles_gateway_errors(tmp_path: Path) -> None:
    """Gateway errors should result in an empty module list."""
    plugin = CoverageIngestPlugin()
    failing_gateway = FailingGateway("db down")
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(
            gateway=cast("StorageGateway", failing_gateway),
            resources=TargetResourceOverrides(modules=()),
        ),
    )

    paths = get_module_paths(ctx)

    expect_equal(paths, [])


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
    repo_cov = repo_root / ".coverage"
    repo_cov.write_text("binary-ish", encoding="utf-8")

    resolved = resolve_coverage_file(ctx)

    expect_equal(resolved, repo_cov)


@pytest.mark.anyio
async def test_execute_ingests_coverage_with_fake_collector(tmp_path: Path) -> None:
    """Happy path: coverage rows are written using the fake collector."""
    plugin = CoverageIngestPlugin()
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    coverage_file = write_coverage_file(repo_root, filename=".coverage", content="{}")

    fake_providers = FakeProviders()
    fake_providers.coverage_collector.coverage_data = {
        "pkg/mod.py": CoverageData(
            path="pkg/mod.py",
            covered_lines=frozenset({1, 2}),
            missing_lines=frozenset({3}),
        )
    }
    overrides = TargetResourceOverrides(
        providers=cast("Providers", fake_providers),
        modules=("pkg/mod.py",),
    )
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expected_rows = 3
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
    plugin = CoverageIngestPlugin()
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    write_coverage_file(repo_root, filename=".coverage", content="{}")
    overrides = TargetResourceOverrides(providers=None, modules=("pkg/mod.py",))
    caplog.set_level(logging.WARNING)
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is False)
    expect_true("Coverage ingest failed" in (result.error_message or ""))
    expect_true("Coverage collector not available" in (result.error_message or ""))
    expect_true("Coverage export failed" in caplog.text)


@pytest.mark.anyio
async def test_execute_fails_when_collector_raises(tmp_path: Path) -> None:
    """Collector exceptions propagate as failed plugin results."""
    plugin = CoverageIngestPlugin()
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    write_coverage_file(repo_root, filename=".coverage", content="{}")

    failing_providers = FakeProviders()
    failing_providers.coverage_collector = cast("FakeCoverageCollector", _FailingCollector())
    overrides = TargetResourceOverrides(
        providers=cast("Providers", failing_providers),
        modules=("pkg/mod.py",),
    )
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is False)
    expect_true("Coverage ingest failed" in (result.error_message or ""))
