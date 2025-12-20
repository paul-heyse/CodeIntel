"""Tests for authentication enforcement on public interfaces."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.settings import ServingSettings, get_serving_settings
from tests._helpers.assertions.expectation_assertions import (
    expect_false,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextlib.contextmanager
def _set_env(env: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables.

    Parameters
    ----------
    env
        Environment variables to set.

    Yields
    ------
    None
        Context manager scope.
    """
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_fails_without_auth_on_0000(tmp_path: Path) -> None:
    """Verify startup fails without auth when bound to 0.0.0.0."""
    settings = ServingSettings(serve_dir=tmp_path, host="0.0.0.0")  # noqa: S104
    with pytest.raises(ValueError, match="Security error"):
        settings.validate_auth_for_host()


def test_fails_without_auth_on_ipv6_any(tmp_path: Path) -> None:
    """Verify startup fails without auth when bound to ::."""
    settings = ServingSettings(serve_dir=tmp_path, host="::")
    with pytest.raises(ValueError, match="Security error"):
        settings.validate_auth_for_host()


def test_fails_without_auth_on_empty_host(tmp_path: Path) -> None:
    """Verify startup fails without auth when bound to empty host."""
    settings = ServingSettings(serve_dir=tmp_path, host="")
    with pytest.raises(ValueError, match="Security error"):
        settings.validate_auth_for_host()


def test_error_message_mentions_env_vars(tmp_path: Path) -> None:
    """Verify error message includes helpful env var references."""
    settings = ServingSettings(serve_dir=tmp_path, host="0.0.0.0")  # noqa: S104
    with pytest.raises(ValueError, match="CODEINTEL_AUTH_TOKEN"):
        settings.validate_auth_for_host()


def test_error_message_mentions_api_key(tmp_path: Path) -> None:
    """Verify error message mentions api_key option."""
    settings = ServingSettings(serve_dir=tmp_path, host="0.0.0.0")  # noqa: S104
    with pytest.raises(ValueError, match="CODEINTEL_SERVE_API_KEY"):
        settings.validate_auth_for_host()


def test_error_message_mentions_disable_option(tmp_path: Path) -> None:
    """Verify error message mentions how to disable check."""
    settings = ServingSettings(serve_dir=tmp_path, host="0.0.0.0")  # noqa: S104
    with pytest.raises(ValueError, match="CODEINTEL_AUTH_REQUIRED_FOR_REMOTE=0"):
        settings.validate_auth_for_host()


def test_succeeds_with_auth_token_on_0000(tmp_path: Path) -> None:
    """Verify startup succeeds with auth_token when bound to 0.0.0.0."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        host="0.0.0.0",  # noqa: S104
        auth_token="test-token-12345",  # noqa: S106
    )
    # Should not raise
    settings.validate_auth_for_host()


def test_succeeds_with_api_key_on_0000(tmp_path: Path) -> None:
    """Verify startup succeeds with api_key when bound to 0.0.0.0."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        host="0.0.0.0",  # noqa: S104
        api_key="test-api-key-12345",
    )
    # Should not raise
    settings.validate_auth_for_host()


def test_succeeds_with_both_auth_methods(tmp_path: Path) -> None:
    """Verify startup succeeds when both auth methods are set."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        host="0.0.0.0",  # noqa: S104
        auth_token="test-token",  # noqa: S106
        api_key="test-key",
    )
    # Should not raise
    settings.validate_auth_for_host()


def test_localhost_ipv4_no_auth_required(tmp_path: Path) -> None:
    """Verify 127.0.0.1 binding doesn't require auth."""
    settings = ServingSettings(serve_dir=tmp_path, host="127.0.0.1")
    # Should not raise
    settings.validate_auth_for_host()


def test_localhost_name_no_auth_required(tmp_path: Path) -> None:
    """Verify localhost binding doesn't require auth."""
    settings = ServingSettings(serve_dir=tmp_path, host="localhost")
    # Should not raise
    settings.validate_auth_for_host()


def test_localhost_ipv6_no_auth_required(tmp_path: Path) -> None:
    """Verify ::1 binding doesn't require auth."""
    settings = ServingSettings(serve_dir=tmp_path, host="::1")
    # Should not raise
    settings.validate_auth_for_host()


def test_disabled_allows_public_bind_without_auth(tmp_path: Path) -> None:
    """Verify disabled check allows public bind without auth."""
    settings = ServingSettings(
        serve_dir=tmp_path,
        host="0.0.0.0",  # noqa: S104
        auth_required_for_remote=False,
    )
    # Should not raise
    settings.validate_auth_for_host()


def test_disabled_via_env_var(tmp_path: Path) -> None:
    """Verify auth enforcement can be disabled via environment variable."""
    with _set_env(
        {
            "CODEINTEL_AUTH_REQUIRED_FOR_REMOTE": "0",
            "CODEINTEL_HOST": "0.0.0.0",  # noqa: S104
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    ):
        settings = get_serving_settings()
    expect_false(settings.auth_required_for_remote)
    # Should not raise
    settings.validate_auth_for_host()


def test_auth_required_for_remote_default_true(tmp_path: Path) -> None:
    """Verify auth_required_for_remote defaults to True."""
    settings = ServingSettings(serve_dir=tmp_path)
    expect_true(settings.auth_required_for_remote)


def test_auth_required_for_remote_from_env(tmp_path: Path) -> None:
    """Verify auth_required_for_remote loads from environment."""
    with _set_env(
        {
            "CODEINTEL_AUTH_REQUIRED_FOR_REMOTE": "1",
            "CODEINTEL_SERVE_DIR": str(tmp_path),
        }
    ):
        settings = get_serving_settings()
    expect_true(settings.auth_required_for_remote)
