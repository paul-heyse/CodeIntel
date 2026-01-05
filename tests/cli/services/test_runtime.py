"""Tests for RuntimeService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeintel.cli.resolution.errors import ResolutionError
from codeintel.cli.services.params import ParamService
from codeintel.cli.services.runtime import RuntimeService
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_not_none,
    expect_true,
)


def test_from_dict() -> None:
    """Create from dictionary."""
    service = RuntimeService.from_dict({"project_root": Path()})
    expect_is_not_none(service)
    expect_false(service.is_resolved)


def test_from_param_service() -> None:
    """Create from ParamService."""
    params = ParamService({"project_root": Path()})
    service = RuntimeService.from_param_service(params)
    expect_is_not_none(service)


def test_explicit_overrides() -> None:
    """Explicit parameters override params dict."""
    service = RuntimeService(
        {"project_root": Path("/original")},
        project_root=Path("/override"),
    )
    expect_equal(service.params["project_root"], Path("/override"))


def test_is_resolved_initially_false() -> None:
    """Service starts unresolved."""
    service = RuntimeService({})
    expect_false(service.is_resolved)


def test_invalidate_clears_cache() -> None:
    """Invalidate clears cached runtime."""
    service = RuntimeService({})
    with patch.object(RuntimeService, "_resolve", return_value=MagicMock()):
        _ = service.runtime
    expect_true(service.is_resolved)

    service.invalidate()
    expect_false(service.is_resolved)


def test_runtime_property_resolves_lazily(tmp_path: Path) -> None:
    """Runtime property triggers resolution."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db_dir = project_dir / "build" / "db"
    db_dir.mkdir(parents=True)

    config_file = project_dir / "config/codeintel.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

    service = RuntimeService({"project_root": project_dir})
    expect_false(service.is_resolved)

    runtime = service.runtime
    expect_true(service.is_resolved)
    expect_is_not_none(runtime)


def test_db_path_property(tmp_path: Path) -> None:
    """Access db_path through service."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    db_dir = project_dir / "build" / "db"
    db_dir.mkdir(parents=True)

    config_file = project_dir / "config/codeintel.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("repo: test/repo\nstorage:\n  db_path: build/db/codeintel.duckdb\n")

    service = RuntimeService({"project_root": project_dir})
    db_path = service.db_path
    expect_in("codeintel.duckdb", str(db_path))


def test_resolution_error_propagates() -> None:
    """Resolution errors propagate."""
    service = RuntimeService({}, allow_fallback=False)
    with pytest.raises(ResolutionError):
        _ = service.runtime
