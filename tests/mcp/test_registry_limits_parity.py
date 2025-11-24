"""Ensure registry/limits helpers remain consistent across transports."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.query_service import BackendLimits
from codeintel.server.datasets import build_dataset_registry, build_registry_and_limits


def test_limits_parity_between_local_and_remote_configs() -> None:
    """BackendLimits derived from config should be identical across modes when values match."""
    base_cfg = {
        "repo": "r",
        "commit": "c",
        "repo_root": Path.cwd(),
        "default_limit": 123,
        "max_rows_per_call": 456,
    }
    local_cfg = ServingConfig(mode="local_db", db_path=Path(":memory:"), **base_cfg)
    remote_cfg = ServingConfig(mode="remote_api", api_base_url="http://localhost", **base_cfg)

    local_limits = BackendLimits.from_config(local_cfg)
    remote_limits = BackendLimits.from_config(remote_cfg)
    if local_limits != remote_limits:
        pytest.fail(f"Limit mismatch: {local_limits} vs {remote_limits}")


def test_registry_helper_matches_base_registry() -> None:
    """build_registry_and_limits should return the canonical dataset registry."""
    cfg = ServingConfig(
        mode="local_db",
        repo="r",
        commit="c",
        repo_root=Path.cwd(),
        db_path=Path(":memory:"),
    )
    registry, limits = build_registry_and_limits(cfg)
    base_registry = build_dataset_registry()
    if registry != base_registry:
        pytest.fail("Registry mismatch between helper and base builder")
    expected_limits = BackendLimits.from_config(cfg)
    if limits != expected_limits:
        pytest.fail("Limits mismatch between helper and BackendLimits.from_config")
