"""Tests for CodeIntelConfig target normalization."""

from __future__ import annotations

from pathlib import Path

from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from tests._helpers.assertions import expect_equal


def test_default_targets_normalizes_empty() -> None:
    """Defaults should be applied when targets are missing or empty."""
    repo = RepoConfig(repo="org/repo", commit="deadbeef")
    paths = CliPathsInput(repo_root=Path.cwd())
    cfg = CodeIntelConfig.from_cli_args(repo_cfg=repo, paths_cfg=paths, options=CliConfigOptions())

    expect_equal(cfg.default_targets, ["export_docs"])


def test_default_targets_preserves_inputs() -> None:
    """Provided targets should be preserved in order."""
    repo = RepoConfig(repo="org/repo", commit="deadbeef")
    paths = CliPathsInput(repo_root=Path.cwd())
    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo,
        paths_cfg=paths,
        options=CliConfigOptions(default_targets=["foo", "bar"]),
    )

    expect_equal(cfg.default_targets, ["foo", "bar"])
