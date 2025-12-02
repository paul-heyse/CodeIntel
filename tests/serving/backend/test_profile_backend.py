"""Tests for ProfileBackend behavior."""

from __future__ import annotations

import pytest

from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.profile_backend import ProfileBackend
from codeintel.storage.gateway import StorageGateway


def _expect(*, condition: bool, message: str) -> None:
    """Fail the test when a condition is not met."""
    if not condition:
        pytest.fail(message)


def _build_components(
    gateway: StorageGateway, limits: BackendLimits | None = None
) -> tuple[BackendContext, DuckDBRepositories]:
    repo = gateway.config.repo or "demo/repo"
    commit = gateway.config.commit or "deadbeef"
    context = BackendContext(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=limits or BackendLimits(),
        graph_engine=None,
    )
    repositories = DuckDBRepositories(gateway, context.repo, context.commit)
    return context, repositories


def test_get_file_profile_not_found(architecture_gateway: StorageGateway) -> None:
    """Return not-found metadata when file profile is missing."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileBackend(context=context, repositories=repositories)

    result = backend.get_file_profile(rel_path="nonexistent.py")

    _expect(condition=result.found is False, message="Missing profiles should be marked not found")
    _expect(condition=result.profile is None, message="Profile payload should be None when missing")
    _expect(
        condition=any(message.code == "not_found" for message in result.meta.messages),
        message="Metadata should include not_found code",
    )


def test_get_module_architecture_success(architecture_gateway: StorageGateway) -> None:
    """Return module architecture for seeded module."""
    context, repositories = _build_components(architecture_gateway)
    backend = ProfileBackend(context=context, repositories=repositories)

    result = backend.get_module_architecture(module="pkg.mod")

    _expect(condition=result.found is True, message="Architecture should be found")
    if result.architecture is None:
        pytest.fail("Architecture payload should not be None")
    _expect(
        condition=result.architecture.get("module") == "pkg.mod",
        message="Module name should be preserved",
    )
