"""Tests for StorageService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.services.runtime import RuntimeService
from codeintel.cli.services.storage import StorageService
from tests._helpers.assertions import expect_equal, expect_false, expect_true

# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_from_path(tmp_path: Path) -> None:
    """Create from explicit path."""
    db_path = tmp_path / "test.duckdb"
    service = StorageService.from_path(db_path)
    expect_equal(service.db_path, db_path)


def test_from_runtime() -> None:
    """Create from RuntimeService."""
    runtime = MagicMock(spec=RuntimeService)
    runtime.db_path = Path("test.duckdb")
    service = StorageService.from_runtime(runtime)
    expect_equal(service.db_path, Path("test.duckdb"))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_is_open_initially_false(tmp_path: Path) -> None:
    """Service starts with no gateway."""
    service = StorageService.from_path(tmp_path / "test.duckdb")
    expect_false(service.is_open)


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Close can be called multiple times."""
    service = StorageService.from_path(tmp_path / "test.duckdb")
    service.close()
    service.close()


def test_gateway_raises_after_close(tmp_path: Path) -> None:
    """Gateway access raises after close."""
    service = StorageService.from_path(tmp_path / "test.duckdb")
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = service.gateway


def test_context_manager_closes(tmp_path: Path) -> None:
    """Context manager closes service on exit."""
    with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
        mock_gateway = MagicMock()
        mock_open.return_value = mock_gateway

        service = StorageService.from_path(tmp_path / "test.duckdb")
        with service:
            _ = service.gateway
            expect_true(service.is_open)
        expect_false(service.is_open)
        mock_gateway.close.assert_called_once()


# ---------------------------------------------------------------------------
# Gateway access
# ---------------------------------------------------------------------------


def test_gateway_opens_lazily(tmp_path: Path) -> None:
    """Gateway opens on first access."""
    db_path = tmp_path / "test.duckdb"

    with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
        mock_gateway = MagicMock()
        mock_open.return_value = mock_gateway

        service = StorageService.from_path(db_path)
        expect_false(service.is_open)

        gateway = service.gateway
        expect_true(service.is_open)
        expect_true(gateway is mock_gateway)
        mock_open.assert_called_once()


def test_gateway_cached(tmp_path: Path) -> None:
    """Gateway is cached after first access."""
    db_path = tmp_path / "test.duckdb"

    with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
        mock_gateway = MagicMock()
        mock_open.return_value = mock_gateway

        service = StorageService.from_path(db_path)
        gateway1 = service.gateway
        gateway2 = service.gateway

        expect_true(gateway1 is gateway2)
        mock_open.assert_called_once()


def test_gateway_scope_read_only(tmp_path: Path) -> None:
    """Gateway scope opens read-only by default."""
    db_path = tmp_path / "test.duckdb"

    with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
        mock_gateway = MagicMock()
        mock_open.return_value = mock_gateway

        service = StorageService.from_path(db_path)
        with service.gateway_scope() as gw:
            expect_true(gw is mock_gateway)

        mock_gateway.close.assert_called_once()


def test_write_gateway_opens_writable(tmp_path: Path) -> None:
    """Write gateway opens with read_only=False."""
    db_path = tmp_path / "test.duckdb"

    with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
        mock_gateway = MagicMock()
        mock_open.return_value = mock_gateway

        service = StorageService.from_path(db_path)
        with service.write_gateway():
            pass

        call_args = mock_open.call_args
        config = call_args[0][0]
        expect_false(config.read_only)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_db_path_from_explicit(tmp_path: Path) -> None:
    """Use explicit db_path."""
    db_path = tmp_path / "explicit.duckdb"
    service = StorageService(db_path=db_path)
    expect_equal(service.db_path, db_path)


def test_db_path_from_runtime() -> None:
    """Use runtime db_path."""
    runtime = MagicMock(spec=RuntimeService)
    runtime.db_path = Path("/from/runtime.duckdb")
    service = StorageService(runtime=runtime)
    expect_equal(service.db_path, Path("/from/runtime.duckdb"))


def test_db_path_raises_without_source() -> None:
    """Raise error when no path available."""
    service = StorageService()
    with pytest.raises(RuntimeError, match="No database path"):
        _ = service.db_path
