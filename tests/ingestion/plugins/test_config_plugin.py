"""Tests for ConfigIngestPlugin wiring and error handling."""

from __future__ import annotations

import pytest

from codeintel.ingestion.plugins.config_plugin import ConfigIngestPlugin
from tests.ingestion.test_runner_plumbing import (
    build_repo_with_configs,
    build_target_context_for_plugin,
)


def _assert_config_rows(ctx) -> int:
    row = ctx.gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.config_values WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    return int(row[0]) if row else 0


@pytest.mark.anyio
async def test_execute_with_no_config_files_returns_empty_result(tmp_path) -> None:
    """When no config files are found, plugin should succeed with no rows."""
    plugin = ConfigIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path)

    result = await plugin.execute(ctx)

    assert result.success is True
    assert result.row_counts == {}
    assert _assert_config_rows(ctx) == 0


@pytest.mark.anyio
async def test_execute_ingests_valid_configs_and_logs_invalid(tmp_path, caplog) -> None:
    """Valid configs are ingested while invalid files only emit warnings."""
    repo_root, _ = build_repo_with_configs(tmp_path, include_invalid=True)
    plugin = ConfigIngestPlugin()
    ctx = build_target_context_for_plugin(plugin, tmp_path, repo_root=repo_root)

    result = await plugin.execute(ctx)

    assert result.success is True
    ingested_rows = _assert_config_rows(ctx)
    assert ingested_rows >= 5  # yaml + toml + ini flatten multiple keys
    assert result.row_counts.get("analytics.config_values") == ingested_rows
    assert "Config parse warning" in caplog.text


@pytest.mark.anyio
async def test_execute_only_invalid_configs_fails(tmp_path) -> None:
    """If all configs fail to parse, plugin should fail with an error message."""
    plugin = ConfigIngestPlugin()
    repo_root = tmp_path / "repo"
    broken = repo_root / "config"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "bad.yaml").write_text(":\n  - nope\n", encoding="utf-8")
    ctx = build_target_context_for_plugin(plugin, tmp_path, repo_root=repo_root)

    result = await plugin.execute(ctx)

    assert result.success is False
    assert result.error_message is not None
    assert "bad.yaml" in result.error_message
    assert _assert_config_rows(ctx) == 0
