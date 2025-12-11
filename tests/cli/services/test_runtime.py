"""Tests for RuntimeService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.services.params import ParamService
from codeintel.cli.services.runtime import RuntimeService


class TestRuntimeServiceCreation:
    """Test RuntimeService creation."""

    def test_from_dict(self) -> None:
        """Create from dictionary."""
        service = RuntimeService.from_dict({"project_root": Path(".")})
        assert service is not None
        assert not service.is_resolved

    def test_from_param_service(self) -> None:
        """Create from ParamService."""
        params = ParamService({"project_root": Path(".")})
        service = RuntimeService.from_param_service(params)
        assert service is not None

    def test_explicit_overrides(self) -> None:
        """Explicit parameters override params dict."""
        service = RuntimeService(
            {"project_root": Path("/original")},
            project_root=Path("/override"),
        )
        assert service._params["project_root"] == Path("/override")


class TestRuntimeServiceCaching:
    """Test RuntimeService caching behavior."""

    def test_is_resolved_initially_false(self) -> None:
        """Service starts unresolved."""
        service = RuntimeService({})
        assert not service.is_resolved

    def test_invalidate_clears_cache(self) -> None:
        """Invalidate clears cached runtime."""
        service = RuntimeService({})
        # Manually set the cached value
        service._resolved = MagicMock()
        assert service.is_resolved

        service.invalidate()
        assert not service.is_resolved


class TestRuntimeServiceResolution:
    """Test RuntimeService resolution."""

    def test_runtime_property_resolves_lazily(self, tmp_path: Path) -> None:
        """Runtime property triggers resolution."""
        # Create a minimal project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        db_dir = project_dir / "build" / "db"
        db_dir.mkdir(parents=True)

        config_file = project_dir / "codeintel.yaml"
        config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

        service = RuntimeService({"project_root": project_dir})
        assert not service.is_resolved

        runtime = service.runtime
        assert service.is_resolved
        assert runtime is not None

    def test_db_path_property(self, tmp_path: Path) -> None:
        """Access db_path through service."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        db_dir = project_dir / "build" / "db"
        db_dir.mkdir(parents=True)

        config_file = project_dir / "codeintel.yaml"
        config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

        service = RuntimeService({"project_root": project_dir})
        db_path = service.db_path
        assert "codeintel.duckdb" in str(db_path)

    def test_resolution_error_propagates(self) -> None:
        """Resolution errors propagate."""
        # No project file, no fallback params
        service = RuntimeService({}, allow_fallback=False)
        with pytest.raises(ResolutionError):
            _ = service.runtime
