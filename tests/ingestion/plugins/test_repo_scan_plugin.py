"""Tests for RepoScanPlugin integration and row count computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.build.contracts import OutputContract
from codeintel.build.targets import OutputTarget
from codeintel.config.datasets.primitives import Column, TableSchema
from codeintel.ingestion.plugins.repo_scan import RepoScanPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.contexts import EnvOverrides, ExecutionContextBuilder

MODULE_COUNT_WITH_INIT = 3


def _make_target(plugin: RepoScanPlugin, tables: tuple[TableSchema, ...]) -> OutputTarget:
    return OutputTarget(
        name=plugin.plugin_name,
        module="ingestion",
        plugin=plugin.plugin_name,
        contract=OutputContract(tables=tables),
        description=plugin.plugin_description,
    )


def _create_context(
    plugin: RepoScanPlugin,
    tmp_path: Path,
    repo_root: Path,
    tables: tuple[TableSchema, ...],
) -> tuple[OutputTarget, TargetExecutionContext]:
    overrides = EnvOverrides(tmp_path=repo_root)
    builder = ExecutionContextBuilder.create(tmp_path, env_overrides=overrides)
    target = _make_target(plugin, tables)
    ctx = builder.build_target_context(target=target)
    return target, ctx


@pytest.mark.anyio
async def test_execute_populates_modules_and_repo_map(tmp_path: Path) -> None:
    """Repo scan should write modules, create change tracker, and populate repo_map."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "x = 1\n",
            "pkg/b.py": "y = 2\n",
        },
    )
    plugin = RepoScanPlugin()
    tables = (
        TableSchema("core", "modules", [Column("module", "VARCHAR")]),
        TableSchema("core", "repo_map", [Column("repo", "VARCHAR")]),
    )
    _target, ctx = _create_context(plugin, tmp_path, repo_root, tables)

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_true(ctx.resources.change_tracker is not None)

    paths = {
        row[0]
        for row in ctx.gateway.con.execute(
            "SELECT path FROM core.modules WHERE repo = ? AND commit = ? ORDER BY path",
            [DEFAULT_REPO, DEFAULT_COMMIT],
        ).fetchall()
    }
    expect_equal(paths, {"pkg/__init__.py", "pkg/a.py", "pkg/b.py"})

    repo_map_row = ctx.gateway.con.execute(
        "SELECT modules, overlays FROM core.repo_map WHERE repo = ? AND commit = ?",
        [DEFAULT_REPO, DEFAULT_COMMIT],
    ).fetchone()
    if repo_map_row is None:
        pytest.fail("repo_map row should be written for scanned repo")
    modules_json, overlays_json = repo_map_row
    modules_list = json.loads(modules_json)
    expect_equal(len(modules_list), MODULE_COUNT_WITH_INIT)
    expect_true(all("pkg" in entry for entry in modules_list))
    expect_equal(json.loads(overlays_json), {})

    expect_equal(result.row_counts.get("core.modules"), MODULE_COUNT_WITH_INIT)
    expect_equal(result.row_counts.get("core.repo_map"), 1)


@pytest.mark.anyio
async def test_compute_row_counts_handles_missing_table(tmp_path: Path) -> None:
    """Row count computation should return 0 for tables that are absent."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/mod.py": "x = 1\n"})
    plugin = RepoScanPlugin()
    tables = (
        TableSchema("core", "modules", [Column("module", "VARCHAR")]),
        TableSchema("core", "missing_table", [Column("id", "INTEGER")]),
    )
    _target, ctx = _create_context(plugin, tmp_path, repo_root, tables)

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts.get("core.modules"), 1)
    expect_equal(result.row_counts.get("core.missing_table"), 0)


@pytest.mark.anyio
async def test_row_counts_ignore_absent_tables_without_errors(tmp_path: Path) -> None:
    """Missing tables should not raise and should return zero counts."""
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/only.py": "y = 2\n"})
    plugin = RepoScanPlugin()
    tables = (
        TableSchema("core", "modules", [Column("module", "VARCHAR")]),
        TableSchema("core", "absent_table", [Column("id", "INTEGER")]),
    )
    _target, ctx = _create_context(plugin, tmp_path, repo_root, tables)

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts.get("core.modules"), 1)
    expect_equal(result.row_counts.get("core.absent_table"), 0)
