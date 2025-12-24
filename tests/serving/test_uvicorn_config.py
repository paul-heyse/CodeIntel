"""Tests for Uvicorn production configuration settings."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.settings import ServingSettings, get_serving_settings
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_none,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

# Test constants
DEFAULT_WORKERS = 1
DEFAULT_TIMEOUT_KEEP_ALIVE = 30
DEFAULT_BACKLOG = 2048
OVERRIDE_WORKERS = 4
OVERRIDE_LIMIT_CONCURRENCY = 100
OVERRIDE_LIMIT_MAX_REQUESTS = 10000
OVERRIDE_TIMEOUT_KEEP_ALIVE = 60
OVERRIDE_BACKLOG = 4096
PRODUCTION_WORKERS = 4
PRODUCTION_LIMIT_CONCURRENCY = 500
PRODUCTION_LIMIT_MAX_REQUESTS = 10000
PRODUCTION_TIMEOUT_KEEP_ALIVE = 60
PRODUCTION_BACKLOG = 4096

pytestmark = pytest.mark.usefixtures("codeintel_env")


def test_uvicorn_workers_default(tmp_path: Path) -> None:
    """Verify default workers is 1."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_equal(settings.uvicorn_workers, DEFAULT_WORKERS)


def test_uvicorn_loop_default(tmp_path: Path) -> None:
    """Verify default event loop is auto."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_equal(settings.uvicorn_loop, "auto")


def test_uvicorn_http_default(tmp_path: Path) -> None:
    """Verify default HTTP implementation is auto."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_equal(settings.uvicorn_http, "auto")


def test_uvicorn_limit_concurrency_default(tmp_path: Path) -> None:
    """Verify default concurrency limit is None (unlimited)."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_is_none(settings.uvicorn_limit_concurrency)


def test_uvicorn_limit_max_requests_default(tmp_path: Path) -> None:
    """Verify default max requests is None (unlimited)."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_is_none(settings.uvicorn_limit_max_requests)


def test_uvicorn_timeout_keep_alive_default(tmp_path: Path) -> None:
    """Verify default keep-alive timeout is 30 seconds."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_equal(settings.uvicorn_timeout_keep_alive, DEFAULT_TIMEOUT_KEEP_ALIVE)


def test_uvicorn_backlog_default(tmp_path: Path) -> None:
    """Verify default backlog is 2048."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_equal(settings.uvicorn_backlog, DEFAULT_BACKLOG)


def test_uvicorn_access_log_default(tmp_path: Path) -> None:
    """Verify access logging is enabled by default."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_true(settings.uvicorn_access_log)


def test_uvicorn_server_header_default(tmp_path: Path) -> None:
    """Verify server header is disabled by default for security."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_false(settings.uvicorn_server_header)


def test_uvicorn_proxy_headers_default(tmp_path: Path) -> None:
    """Verify proxy headers are disabled by default."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_false(settings.uvicorn_proxy_headers)


def test_uvicorn_forwarded_allow_ips_default(tmp_path: Path) -> None:
    """Verify default forwarded allow IPs is localhost."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_equal(settings.uvicorn_forwarded_allow_ips, "127.0.0.1")


def test_uvicorn_workers_from_env(tmp_path: Path) -> None:
    """Verify workers setting loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_WORKERS": str(OVERRIDE_WORKERS),
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_workers, OVERRIDE_WORKERS)


def test_uvicorn_loop_from_env(tmp_path: Path) -> None:
    """Verify event loop setting loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_LOOP": "uvloop",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_loop, "uvloop")


def test_uvicorn_http_from_env(tmp_path: Path) -> None:
    """Verify HTTP implementation setting loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_HTTP": "httptools",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_http, "httptools")


def test_uvicorn_limit_concurrency_from_env(tmp_path: Path) -> None:
    """Verify concurrency limit loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_LIMIT_CONCURRENCY": str(OVERRIDE_LIMIT_CONCURRENCY),
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_limit_concurrency, OVERRIDE_LIMIT_CONCURRENCY)


def test_uvicorn_limit_concurrency_empty_is_none(tmp_path: Path) -> None:
    """Verify empty concurrency limit string becomes None."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_LIMIT_CONCURRENCY": "",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_is_none(settings.uvicorn_limit_concurrency)


def test_uvicorn_limit_max_requests_from_env(tmp_path: Path) -> None:
    """Verify max requests loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_LIMIT_MAX_REQUESTS": str(OVERRIDE_LIMIT_MAX_REQUESTS),
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_limit_max_requests, OVERRIDE_LIMIT_MAX_REQUESTS)


def test_uvicorn_timeout_keep_alive_from_env(tmp_path: Path) -> None:
    """Verify keep-alive timeout loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_TIMEOUT_KEEP_ALIVE": str(OVERRIDE_TIMEOUT_KEEP_ALIVE),
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_timeout_keep_alive, OVERRIDE_TIMEOUT_KEEP_ALIVE)


def test_uvicorn_backlog_from_env(tmp_path: Path) -> None:
    """Verify backlog loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_BACKLOG": str(OVERRIDE_BACKLOG),
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_backlog, OVERRIDE_BACKLOG)


def test_uvicorn_access_log_disabled_from_env(tmp_path: Path) -> None:
    """Verify access log can be disabled via environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_ACCESS_LOG": "0",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_false(settings.uvicorn_access_log)


def test_uvicorn_server_header_enabled_from_env(tmp_path: Path) -> None:
    """Verify server header can be enabled via environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_SERVER_HEADER": "1",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_true(settings.uvicorn_server_header)


def test_uvicorn_proxy_headers_enabled_from_env(tmp_path: Path) -> None:
    """Verify proxy headers can be enabled via environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_PROXY_HEADERS": "1",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_true(settings.uvicorn_proxy_headers)


def test_uvicorn_forwarded_allow_ips_from_env(tmp_path: Path) -> None:
    """Verify forwarded allow IPs loads from environment."""
    os.environ.update(
        {
            "CODEINTEL_UVICORN_FORWARDED_ALLOW_IPS": "10.0.0.0/8",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    )
    settings = get_serving_settings()
    expect_equal(settings.uvicorn_forwarded_allow_ips, "10.0.0.0/8")


def test_production_config_with_workers(tmp_path: Path) -> None:
    """Verify production config with multiple workers."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        uvicorn_workers=PRODUCTION_WORKERS,
        uvicorn_loop="uvloop",
        uvicorn_http="httptools",
        uvicorn_server_header=False,
        uvicorn_proxy_headers=True,
        uvicorn_forwarded_allow_ips="127.0.0.1,10.0.0.0/8",
    )
    expect_equal(settings.uvicorn_workers, PRODUCTION_WORKERS)
    expect_equal(settings.uvicorn_loop, "uvloop")
    expect_equal(settings.uvicorn_http, "httptools")
    expect_false(settings.uvicorn_server_header)
    expect_true(settings.uvicorn_proxy_headers)
    expect_true("10.0.0.0/8" in settings.uvicorn_forwarded_allow_ips)


def test_production_config_with_limits(tmp_path: Path) -> None:
    """Verify production config with resource limits."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        uvicorn_limit_concurrency=PRODUCTION_LIMIT_CONCURRENCY,
        uvicorn_limit_max_requests=PRODUCTION_LIMIT_MAX_REQUESTS,
        uvicorn_timeout_keep_alive=PRODUCTION_TIMEOUT_KEEP_ALIVE,
        uvicorn_backlog=PRODUCTION_BACKLOG,
    )
    expect_equal(settings.uvicorn_limit_concurrency, PRODUCTION_LIMIT_CONCURRENCY)
    expect_equal(settings.uvicorn_limit_max_requests, PRODUCTION_LIMIT_MAX_REQUESTS)
    expect_equal(settings.uvicorn_timeout_keep_alive, PRODUCTION_TIMEOUT_KEEP_ALIVE)
    expect_equal(settings.uvicorn_backlog, PRODUCTION_BACKLOG)
