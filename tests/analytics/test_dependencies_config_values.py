"""Regression tests for config_values joins in dependency analytics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.analytics.dependencies import load_config_key_map
from codeintel.config.schemas.sql_builder import ensure_schema
from tests._helpers.fixtures import provision_gateway_with_repo


def test_load_config_keys_filters_repo(tmp_path: Path) -> None:
    """Ensure config_values rows are filtered by repo/commit."""
    repo_root = tmp_path / "repo"
    with provision_gateway_with_repo(repo_root) as ctx:
        ensure_schema(ctx.gateway.con, "analytics.config_values")
        ctx.gateway.con.executemany(
            """
            INSERT INTO analytics.config_values (
                repo, commit, config_path, format, key, reference_paths, reference_modules, reference_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    ctx.repo,
                    ctx.commit,
                    "cfg/app.yaml",
                    "yaml",
                    "feature.flag",
                    json.dumps(["cfg/app.yaml"]),
                    json.dumps(["pkg.mod"]),
                    1,
                ),
                (
                    "other/repo",
                    "deadbeef",
                    "cfg/other.yaml",
                    "yaml",
                    "feature.flag",
                    json.dumps(["cfg/other.yaml"]),
                    json.dumps(["other.mod"]),
                    1,
                ),
            ],
        )

        mapping = load_config_key_map(ctx.gateway.con, ctx.repo, ctx.commit)

        if mapping != {"pkg.mod": {"feature.flag"}}:
            pytest.fail(f"Unexpected config key mapping: {mapping}")
