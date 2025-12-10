"""Tests for ConfigIngestPlugin wiring and error handling."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from codeintel.build.context import TargetExecutionContext
from codeintel.build.result import TargetResult
from codeintel.ingestion.plugins.config_plugin import ConfigIngestPlugin
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.assertions.logging_assertions import assert_logged
from tests._helpers.fakes.ingestion_context import build_repo_tree
from tests._helpers.ingestion import (
    TargetContextConfig,
    build_repo_with_configs,
    repo_variants,
    run_ingestion_scenario,
)

# Minimum rows expected from flattening yaml + toml + ini config values
MIN_CONFIG_ROWS_EXPECTED = 5


def _assert_config_rows(ctx: TargetExecutionContext) -> int:
    row = ctx.gateway.con.execute(
        "SELECT COUNT(*) FROM analytics.config_values WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    ).fetchone()
    return int(row[0]) if row else 0


@pytest.mark.anyio
async def test_execute_with_no_config_files_returns_empty_result(tmp_path: Path) -> None:
    """When no config files are found, plugin should succeed with no rows."""
    ctx, raw_result = await run_ingestion_scenario(ConfigIngestPlugin, tmp_path)
    result = cast("TargetResult", raw_result)

    expect_true(result.success is True)
    expect_equal(result.row_counts, {})
    expect_equal(_assert_config_rows(ctx), 0)


@pytest.mark.anyio
async def test_execute_ingests_valid_configs_and_logs_invalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Valid configs are ingested while invalid files only emit warnings."""
    repo_root, _ = build_repo_with_configs(tmp_path, include_invalid=True)
    caplog.set_level("WARNING")
    ctx, raw_result = await run_ingestion_scenario(
        ConfigIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root),
    )
    result = cast("TargetResult", raw_result)

    expect_true(result.success is True)
    ingested_rows = _assert_config_rows(ctx)
    expect_true(ingested_rows >= MIN_CONFIG_ROWS_EXPECTED)
    expect_equal(result.row_counts.get("analytics.config_values"), ingested_rows)
    expect_in("Config parse warning", caplog.text)
    assert_logged(caplog.records, level="WARNING", containing="Config parse warning")


@pytest.mark.anyio
async def test_execute_only_invalid_configs_fails(tmp_path: Path) -> None:
    """If all configs fail to parse, plugin should fail with an error message."""
    variants = repo_variants(
        base_structure={},
        invalid_structure={"config/bad.yaml": ":\n  - nope\n"},
        macro_structure={},
    )
    invalid_structure = variants["with_invalid"].repo_structure or {}
    repo_root = build_repo_tree(tmp_path / "repo", invalid_structure)
    ctx, raw_result = await run_ingestion_scenario(
        ConfigIngestPlugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root),
    )
    result = cast("TargetResult", raw_result)

    expect_true(result.success is False)
    error_message = expect_is_not_none(result.error_message)
    expect_in("bad.yaml", error_message)
    expect_equal(_assert_config_rows(ctx), 0)
