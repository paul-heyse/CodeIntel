"""Tests for RepoScanPlugin integration and row count computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.config.datasets.primitives import Column, TableSchema
from codeintel.ingestion.plugins.repo_scan import RepoScanPlugin
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_repo_target,
    build_repo_tree,
    build_target_context_for_plugin,
)

MODULE_COUNT_WITH_INIT = 3


@pytest.mark.anyio
async def test_execute_populates_modules_and_repo_map(tmp_path: Path, ingestion_gateway) -> None:
    """Repo scan should write modules, create change tracker, and populate repo_map."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "x = 1\n",
            "pkg/b.py": "y = 2\n",
            "pkg/nested/c.py": "z = 3\n",
            "pkg/nested/d.py": "u = 4\n",
        },
    )
    plugin = RepoScanPlugin()
    tables = (
        TableSchema("core", "modules", [Column("module", "VARCHAR")]),
        TableSchema("core", "repo_map", [Column("repo", "VARCHAR")]),
    )
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(
            repo_root=repo_root,
            gateway=ingestion_gateway,
        ),
        target=build_repo_target(plugin, tables),
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_true(ctx.resources.change_tracker is not None)

    modules = {
        row[0]
        for row in ctx.gateway.con.execute(
            "SELECT module FROM core.modules WHERE repo = ? AND commit = ? ORDER BY module",
            [DEFAULT_REPO, DEFAULT_COMMIT],
        ).fetchall()
    }
    expect_equal(modules, {"pkg.__init__", "pkg.a", "pkg.b", "pkg.nested.c", "pkg.nested.d"})

    repo_map_row = ctx.gateway.con.execute(
        "SELECT modules, overlays FROM core.repo_map WHERE repo = ? AND commit = ?",
        [DEFAULT_REPO, DEFAULT_COMMIT],
    ).fetchone()
    if repo_map_row is None:
        pytest.fail("repo_map row should be written for scanned repo")
    modules_json, overlays_json = repo_map_row
    modules_list = json.loads(modules_json)
    expect_equal(len(modules_list), MODULE_COUNT_WITH_INIT + 2)
    expect_equal(json.loads(overlays_json), {})

    expect_equal(result.row_counts.get("core.modules"), MODULE_COUNT_WITH_INIT + 2)
    expect_equal(result.row_counts.get("core.repo_map"), 1)


@pytest.mark.anyio
async def test_compute_row_counts_handles_missing_table(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ingestion_gateway
) -> None:
    """Row count computation should return 0 for tables that are absent."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/mod.py": "x = 1\n", "pkg/nested/mod2.py": "y = 2\n"},
    )
    plugin = RepoScanPlugin()
    caplog.set_level("WARNING")
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(
            repo_root=repo_root,
            gateway=ingestion_gateway,
        ),
        target=build_repo_target(
            plugin,
            (
                TableSchema("core", "modules", [Column("module", "VARCHAR")]),
                TableSchema("core", "missing_table", [Column("id", "INTEGER")]),
            ),
        ),
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts.get("core.modules"), 2)
    expect_equal(result.row_counts.get("core.missing_table"), 0)
    assert_logged(
        caplog.records, level="WARNING", containing="Row count fallback for core.missing_table"
    )


@pytest.mark.anyio
async def test_row_counts_ignore_absent_tables_without_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ingestion_gateway
) -> None:
    """Missing tables should not raise and should return zero counts."""
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/only.py": "y = 2\n", "pkg/deep/nested.py": "z = 3\n"},
    )
    plugin = RepoScanPlugin()
    caplog.set_level("WARNING")
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(
            repo_root=repo_root,
            gateway=ingestion_gateway,
        ),
        target=build_repo_target(
            plugin,
            (
                TableSchema("core", "modules", [Column("module", "VARCHAR")]),
                TableSchema("core", "absent_table", [Column("id", "INTEGER")]),
            ),
        ),
    )

    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_equal(result.row_counts.get("core.modules"), 2)
    expect_equal(result.row_counts.get("core.absent_table"), 0)
    assert_logged(caplog.records, level="WARNING", containing="core.absent_table")
