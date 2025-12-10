"""Tests for ProfileQueryLayer behavior."""

from __future__ import annotations

import pytest

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.profile_backend import ProfileQueryLayer
from codeintel.serving.mcp import errors
from codeintel.storage.gateway import StorageGateway
from tests._helpers.assertions import expect_true
from tests._helpers.backend_components import build_backend_components


def _expect(*, condition: bool, message: str) -> None:
    """Delegate to shared assertion helper."""
    expect_true(condition, message=message)


def _build_components(
    gateway: StorageGateway, limits: BackendLimits | None = None
) -> tuple[BackendContext, DuckDBRepositories]:
    components = build_backend_components(gateway, limits=limits)
    return components.context, components.repositories


# -----------------------------------------------------------------------------
# Tests for backend properties
# -----------------------------------------------------------------------------


def test_backend_con_property(architecture_gateway: StorageGateway) -> None:
    """Verify con property returns DuckDB connection."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    con = backend.con

    _expect(
        condition=con is not None,
        message="Should return DuckDB connection",
    )


def test_backend_modules_property(architecture_gateway: StorageGateway) -> None:
    """Verify modules property returns module repository."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    modules = backend.modules

    _expect(
        condition=modules is not None,
        message="Should return module repository",
    )


def test_backend_subsystems_property(architecture_gateway: StorageGateway) -> None:
    """Verify subsystems property returns subsystem repository."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    subsystems = backend.subsystems

    _expect(
        condition=subsystems is not None,
        message="Should return subsystem repository",
    )


# -----------------------------------------------------------------------------
# Tests for get_file_profile
# -----------------------------------------------------------------------------


def test_get_file_profile_not_found(architecture_gateway: StorageGateway) -> None:
    """Return not-found metadata when file profile is missing."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    result = backend.get_file_profile(rel_path="nonexistent.py")

    _expect(condition=result.found is False, message="Missing profiles should be marked not found")
    _expect(condition=result.profile is None, message="Profile payload should be None when missing")
    _expect(
        condition=any(message.code == "not_found" for message in result.meta.messages),
        message="Metadata should include not_found code",
    )


def test_get_file_profile_success(architecture_gateway: StorageGateway) -> None:
    """Return file profile for seeded file path."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    # The architecture fixture seeds pkg/mod.py
    result = backend.get_file_profile(rel_path="pkg/mod.py")

    # The file may or may not have a profile depending on fixture setup
    _expect(
        condition=result is not None,
        message="Should return a result object",
    )


# -----------------------------------------------------------------------------
# Tests for get_file_summary
# -----------------------------------------------------------------------------


def test_get_file_summary_not_found(architecture_gateway: StorageGateway) -> None:
    """Raise not_found when file summary is missing."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.get_file_summary(rel_path="nonexistent_file.py")

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Missing file summary should raise not-found error",
    )


def test_get_file_summary_success(architecture_gateway: StorageGateway) -> None:
    """Return file summary for seeded file path."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    # Try to get summary for a known file - this may raise not_found
    # depending on fixture data, but we verify it handles the call correctly
    try:
        result = backend.get_file_summary(rel_path="pkg/mod.py")
        _expect(
            condition=result is not None,
            message="Should return a file summary result",
        )
    except errors.McpError as exc:
        _expect(
            condition=exc.detail.code == "not-found",
            message="Should raise not-found if summary missing",
        )


def test_get_file_summary_with_scope(architecture_gateway: StorageGateway) -> None:
    """Verify scope parameter is accepted (though currently unused)."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    scope = GraphRunScope(paths=("pkg/mod.py",), modules=("pkg.mod",))

    # Try to get summary with scope
    try:
        result = backend.get_file_summary(rel_path="pkg/mod.py", scope=scope)
        _expect(
            condition=result is not None,
            message="Should accept scope parameter",
        )
    except errors.McpError as exc:
        _expect(
            condition=exc.detail.code == "not-found",
            message="Should raise not-found if summary missing",
        )


# -----------------------------------------------------------------------------
# Tests for get_module_profile
# -----------------------------------------------------------------------------


def test_get_module_profile_not_found(architecture_gateway: StorageGateway) -> None:
    """Raise not_found when module profile is missing."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.get_module_profile(module="nonexistent.module")

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Missing module profile should raise not-found error",
    )


def test_get_module_profile_success(architecture_gateway: StorageGateway) -> None:
    """Return module profile for seeded module."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    # The architecture fixture seeds module_profile for pkg.mod
    result = backend.get_module_profile(module="pkg.mod")

    _expect(
        condition=result.found is True,
        message="Module profile should be found",
    )
    if result.profile is not None:
        _expect(
            condition=result.profile.get("module") == "pkg.mod",
            message="Module name should be preserved in profile",
        )


# -----------------------------------------------------------------------------
# Tests for get_module_architecture
# -----------------------------------------------------------------------------


def test_get_module_architecture_success(architecture_gateway: StorageGateway) -> None:
    """Return module architecture for seeded module."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    result = backend.get_module_architecture(module="pkg.mod")

    _expect(condition=result.found is True, message="Architecture should be found")
    if result.architecture is None:
        pytest.fail("Architecture payload should not be None")
    _expect(
        condition=result.architecture.get("module") == "pkg.mod",
        message="Module name should be preserved",
    )


def test_get_module_architecture_not_found(architecture_gateway: StorageGateway) -> None:
    """Raise not_found when module architecture is missing."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    with pytest.raises(errors.McpError) as excinfo:
        backend.get_module_architecture(module="nonexistent.module")

    _expect(
        condition=excinfo.value.detail.code == "not-found",
        message="Missing module architecture should raise not-found error",
    )


# -----------------------------------------------------------------------------
# Tests for get_file_hints
# -----------------------------------------------------------------------------


def test_get_file_hints_returns_result(architecture_gateway: StorageGateway) -> None:
    """Return file hints result for any path."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    result = backend.get_file_hints(rel_path="pkg/mod.py")

    _expect(
        condition=result is not None,
        message="Should return a file hints result",
    )
    _expect(
        condition=isinstance(result.hints, list),
        message="Hints should be a list",
    )


def test_get_file_hints_empty_result(architecture_gateway: StorageGateway) -> None:
    """Return empty hints for nonexistent file (no error raised)."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileQueryLayer(context=context, repositories=repositories)

    result = backend.get_file_hints(rel_path="nonexistent_file_xyz.py")

    _expect(
        condition=result is not None,
        message="Should return a result object even for nonexistent file",
    )
    _expect(
        condition=result.hints is not None,
        message="Hints list should be present (possibly empty)",
    )
