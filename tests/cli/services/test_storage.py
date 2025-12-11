"""Tests for StorageService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.services.runtime import RuntimeService
from codeintel.cli.services.storage import StorageService


class TestStorageServiceCreation:
    """Test StorageService creation."""

    def test_from_path(self, tmp_path: Path) -> None:
        """Create from explicit path."""
        db_path = tmp_path / "test.duckdb"
        service = StorageService.from_path(db_path)
        assert service.db_path == db_path

    def test_from_runtime(self) -> None:
        """Create from RuntimeService."""
        runtime = MagicMock(spec=RuntimeService)
        runtime.db_path = Path("/tmp/test.duckdb")
        service = StorageService.from_runtime(runtime)
        assert service.db_path == Path("/tmp/test.duckdb")


class TestStorageServiceLifecycle:
    """Test StorageService lifecycle management."""

    def test_is_open_initially_false(self, tmp_path: Path) -> None:
        """Service starts with no gateway."""
        service = StorageService.from_path(tmp_path / "test.duckdb")
        assert not service.is_open

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Close can be called multiple times."""
        service = StorageService.from_path(tmp_path / "test.duckdb")
        service.close()
        service.close()  # Should not raise

    def test_gateway_raises_after_close(self, tmp_path: Path) -> None:
        """Gateway access raises after close."""
        service = StorageService.from_path(tmp_path / "test.duckdb")
        service.close()
        with pytest.raises(RuntimeError, match="closed"):
            _ = service.gateway

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        """Context manager closes service on exit."""
        service = StorageService.from_path(tmp_path / "test.duckdb")
        with service:
            pass
        assert service._closed


class TestStorageServiceGateway:
    """Test StorageService gateway access."""

    def test_gateway_opens_lazily(self, tmp_path: Path) -> None:
        """Gateway opens on first access."""
        db_path = tmp_path / "test.duckdb"

        with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
            mock_gateway = MagicMock()
            mock_open.return_value = mock_gateway

            service = StorageService.from_path(db_path)
            assert not service.is_open

            gateway = service.gateway
            assert service.is_open
            assert gateway is mock_gateway
            mock_open.assert_called_once()

    def test_gateway_cached(self, tmp_path: Path) -> None:
        """Gateway is cached after first access."""
        db_path = tmp_path / "test.duckdb"

        with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
            mock_gateway = MagicMock()
            mock_open.return_value = mock_gateway

            service = StorageService.from_path(db_path)
            gateway1 = service.gateway
            gateway2 = service.gateway

            assert gateway1 is gateway2
            mock_open.assert_called_once()


class TestStorageServiceScopes:
    """Test gateway scope context managers."""

    def test_gateway_scope_read_only(self, tmp_path: Path) -> None:
        """Gateway scope opens read-only by default."""
        db_path = tmp_path / "test.duckdb"

        with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
            mock_gateway = MagicMock()
            mock_open.return_value = mock_gateway

            service = StorageService.from_path(db_path)
            with service.gateway_scope() as gw:
                assert gw is mock_gateway

            mock_gateway.close.assert_called_once()

    def test_write_gateway_opens_writable(self, tmp_path: Path) -> None:
        """Write gateway opens with read_only=False."""
        db_path = tmp_path / "test.duckdb"

        with patch("codeintel.cli.services.storage.open_gateway") as mock_open:
            mock_gateway = MagicMock()
            mock_open.return_value = mock_gateway

            service = StorageService.from_path(db_path)
            with service.write_gateway():
                pass

            # Check that read_only=False was passed
            call_args = mock_open.call_args
            config = call_args[0][0]
            assert config.read_only is False


class TestStorageServiceDbPath:
    """Test database path resolution."""

    def test_db_path_from_explicit(self, tmp_path: Path) -> None:
        """Use explicit db_path."""
        db_path = tmp_path / "explicit.duckdb"
        service = StorageService(db_path=db_path)
        assert service.db_path == db_path

    def test_db_path_from_runtime(self) -> None:
        """Use runtime db_path."""
        runtime = MagicMock(spec=RuntimeService)
        runtime.db_path = Path("/from/runtime.duckdb")
        service = StorageService(runtime=runtime)
        assert service.db_path == Path("/from/runtime.duckdb")

    def test_db_path_raises_without_source(self) -> None:
        """Raise error when no path available."""
        service = StorageService()
        with pytest.raises(RuntimeError, match="No database path"):
            _ = service.db_path
