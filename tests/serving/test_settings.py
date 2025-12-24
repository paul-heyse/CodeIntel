"""Tests for serving settings environment parsing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.settings import get_serving_settings
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_POOL_SIZE = 4
DEFAULT_PORT = 8000
DEFAULT_POLL_INTERVAL_S = 1.0

OVERRIDE_POOL_SIZE = 8
OVERRIDE_PORT = 9000
OVERRIDE_POLL_INTERVAL_S = 0.25

pytestmark = pytest.mark.usefixtures("codeintel_env")


def test_settings_defaults(tmp_path: Path) -> None:
    """Defaults load when env vars are unset."""
    os.environ["CODEINTEL_SERVE_DIR"] = str(tmp_path)
    settings = get_serving_settings()
    expect_equal(settings.serve_dir, tmp_path.resolve())
    expect_true(settings.hot_swap)
    expect_equal(settings.pool_size, DEFAULT_POOL_SIZE)
    expect_equal(settings.poll_interval_s, DEFAULT_POLL_INTERVAL_S)
    expect_equal(settings.mcp_transport, "stdio")
    expect_equal(settings.host, "127.0.0.1")
    expect_equal(settings.port, DEFAULT_PORT)
    expect_is_none(settings.auth_token)


def test_settings_overrides(tmp_path: Path) -> None:
    """Env vars override defaults."""
    expected_auth = "not-a-real-auth-value"
    os.environ.update(
        {
            "CODEINTEL_SERVE_DIR": str(tmp_path / "serve"),
            "CODEINTEL_SERVE_HOTSWAP": "0",
            "CODEINTEL_SERVE_POOL_SIZE": str(OVERRIDE_POOL_SIZE),
            "CODEINTEL_SERVE_POLL_INTERVAL": str(OVERRIDE_POLL_INTERVAL_S),
            "CODEINTEL_MCP_TRANSPORT": "http",
            "CODEINTEL_HOST": "127.0.0.2",
            "CODEINTEL_PORT": str(OVERRIDE_PORT),
            "CODEINTEL_AUTH_TOKEN": expected_auth,
        }
    )
    settings = get_serving_settings()

    expect_equal(settings.serve_dir, (tmp_path / "serve").resolve())
    expect_false(settings.hot_swap)
    expect_equal(settings.pool_size, OVERRIDE_POOL_SIZE)
    expect_equal(settings.poll_interval_s, OVERRIDE_POLL_INTERVAL_S)
    expect_equal(settings.mcp_transport, "http")
    expect_equal(settings.host, "127.0.0.2")
    expect_equal(settings.port, OVERRIDE_PORT)
    expect_equal(expect_is_not_none(settings.auth_token), expected_auth)
