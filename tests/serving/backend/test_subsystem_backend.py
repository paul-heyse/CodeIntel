"""Tests for SubsystemBackend behavior."""

from __future__ import annotations

import pytest

from codeintel.serving.backend import BackendContext, BackendLimits, DuckDBRepositories
from codeintel.serving.backend.subsystem_backend import SubsystemBackend
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


def test_list_subsystems_respects_backend_limits(
    architecture_gateway: StorageGateway,
) -> None:
    """Clamp subsystem listings to backend limits."""
    limits = BackendLimits(default_limit=5, max_rows_per_call=1)
    context, repositories = _build_components(architecture_gateway, limits)
    backend = SubsystemBackend(context=context, repositories=repositories)

    result = backend.list_subsystems(limit=10)

    _expect(condition=result.meta.applied_limit == 1, message="Applied limit should clamp to max")
    _expect(condition=len(result.subsystems) <= 1, message="Result set should respect clamp")


def test_get_module_subsystems_returns_memberships(
    architecture_gateway: StorageGateway,
) -> None:
    """Return subsystem memberships for a module."""
    context, repositories = _build_components(architecture_gateway)
    backend = SubsystemBackend(context=context, repositories=repositories)

    memberships = backend.get_module_subsystems(module="pkg.mod")

    _expect(condition=memberships.found is True, message="Memberships should be marked found")
    _expect(condition=bool(memberships.memberships), message="Membership list should not be empty")
