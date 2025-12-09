"""Tests for ScipIngestPlugin behavior and fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.providers import Providers
from codeintel.ingestion.plugins.scip_plugin import (
    ScipIngestPlugin,
    get_module_paths,
    paths_to_modules,
)
from tests._helpers import DEFAULT_COMMIT, DEFAULT_REPO, build_repo_tree
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.fakes.contexts import TargetResourceOverrides
from tests._helpers.fakes.fake_providers import FakeProviders
from tests._helpers.fakes.recording_gateways import FailingGateway
from tests._helpers.ingestion import TargetContextConfig, build_target_context_for_plugin

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def test_paths_to_modules_creates_records(tmp_path: Path) -> None:
    """Path conversion should map to module names with file paths."""
    repo_root = tmp_path / "repo"
    paths = ["pkg/a.py", "pkg/util/b.py"]
    modules = paths_to_modules(paths, repo_root)

    expect_equal(modules[0].module_name, "pkg.a")
    expect_equal(modules[0].file_path, repo_root / "pkg/a.py")
    expect_equal(modules[1].module_name, "pkg.util.b")


def test_get_module_paths_uses_resources(tmp_path: Path) -> None:
    """resources.modules should be returned directly."""
    plugin = ScipIngestPlugin()
    overrides = TargetResourceOverrides(modules=("pkg/a.py",))
    ctx = build_target_context_for_plugin(
        plugin, tmp_path, config=TargetContextConfig(resources=overrides)
    )
    ctx.gateway.con.execute("DELETE FROM core.modules")

    paths = get_module_paths(ctx)

    expect_equal(paths, ["pkg/a.py"])


def test_get_module_paths_reads_database(tmp_path: Path) -> None:
    """Database rows are used when resources are empty."""
    plugin = ScipIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)
    ctx.gateway.con.execute(
        "INSERT INTO core.modules (module, path, repo, commit) VALUES (?, ?, ?, ?)",
        ["pkg.a", "pkg/a.py", DEFAULT_REPO, DEFAULT_COMMIT],
    )

    paths = get_module_paths(ctx)

    expect_equal(paths, ["pkg/a.py"])


def test_get_module_paths_handles_gateway_error(tmp_path: Path) -> None:
    """Gateway failures should not raise."""
    plugin = ScipIngestPlugin()
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


@pytest.mark.anyio
async def test_execute_raises_when_indexer_missing(tmp_path: Path) -> None:
    """Missing scip_indexer should raise ToolNotAvailableError."""
    plugin = ScipIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)

    with pytest.raises(ToolNotAvailableError):
        await plugin.execute(ctx)


def _write_scip_json(target_dir: Path) -> Path:
    docs = [
        {
            "relativePath": "pkg/a.py",
            "symbols": [{"symbol": "pkg/a.py:func", "documentation": ["doc"]}],
            "occurrences": [
                {
                    "symbol": "pkg/a.py:func",
                    "range": [1, 0, 1, 4],
                    "symbolRoles": 1,
                }
            ],
        }
    ]
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "index.json"
    json_path.write_text(json.dumps({"documents": docs}), encoding="utf-8")
    return json_path


@pytest.mark.anyio
async def test_execute_ingests_symbols_and_occurrences(tmp_path: Path) -> None:
    """SCIP ingestion should write symbols/occurrences and artifacts."""
    plugin = ScipIngestPlugin()
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/a.py": "def a():\n    return 1\n"})
    fake_providers = FakeProviders()
    overrides = TargetResourceOverrides(
        providers=cast("Providers", fake_providers),
        modules=("pkg/a.py",),
    )
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
    )

    _write_scip_json(ctx.scip_dir)
    result = await plugin.execute(ctx)

    expect_true(result.success is True)
    expect_true("index.scip" in result.artifacts_written)
    expect_true("index.json" in result.artifacts_written)
    symbols = ctx.gateway.con.execute(
        "SELECT COUNT(*) FROM core.scip_symbols WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    occurrences = ctx.gateway.con.execute(
        "SELECT COUNT(*) FROM core.scip_occurrences WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    if symbols is None or occurrences is None:
        pytest.fail("SCIP ingestion should write symbols and occurrences")
    expect_true(symbols[0] >= 1)
    expect_true(occurrences[0] >= 1)


@pytest.mark.anyio
async def test_execute_fails_when_indexer_returns_error(tmp_path: Path) -> None:
    """Failed index run should propagate as failed TargetResult."""
    plugin = ScipIngestPlugin()
    repo_root = build_repo_tree(tmp_path / "repo", {"pkg/a.py": "def a():\n    return 1\n"})
    fake_providers = FakeProviders()
    fake_providers.scip_indexer.index_success = False
    overrides = TargetResourceOverrides(
        providers=cast("Providers", fake_providers),
        modules=("pkg/a.py",),
    )
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, resources=overrides),
    )
    _write_scip_json(ctx.scip_dir)

    result = await plugin.execute(ctx)

    expect_true(result.success is False)
    expect_true("SCIP ingest failed" in (result.error_message or ""))
