"""Tests for SubsystemQueryLayer behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.backend import BackendLimits
from codeintel.serving.backend.subsystem_backend import SubsystemQueryLayer
from tests._helpers.assertions import expect_true
from tests._helpers.backend_components import build_backend_components

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def _expect(*, condition: bool, message: str) -> None:
    """Delegate to shared assertion helper."""
    expect_true(condition, message=message)


def test_list_subsystems_respects_backend_limits(
    architecture_gateway: StorageGateway,
) -> None:
    """Clamp subsystem listings to backend limits."""
    limits = BackendLimits(default_limit=5, max_rows_per_call=1)
    components = build_backend_components(architecture_gateway, limits=limits)
    backend = SubsystemQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    result = backend.list_subsystems(limit=10)

    _expect(condition=result.meta.applied_limit == 1, message="Applied limit should clamp to max")
    _expect(condition=len(result.subsystems) <= 1, message="Result set should respect clamp")


def test_get_module_subsystems_returns_memberships(
    architecture_gateway: StorageGateway,
) -> None:
    """Return subsystem memberships for a module."""
    components = build_backend_components(architecture_gateway)
    backend = SubsystemQueryLayer(
        context=components.context,
        repositories=components.repositories,
    )

    memberships = backend.get_module_subsystems(module="pkg.mod")

    _expect(condition=memberships.found is True, message="Memberships should be marked found")
    _expect(condition=bool(memberships.memberships), message="Membership list should not be empty")
