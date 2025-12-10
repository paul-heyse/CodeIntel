"""Shared fixtures for service delegate tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.backend import BackendLimits
from codeintel.storage.gateway import StorageGateway
from tests._helpers.serving_apps import ServiceApp, build_service_app

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


@pytest.fixture
def service_app_factory() -> Callable[..., ServiceApp]:
    """Factory for building service apps bound to a gateway snapshot."""

    def _build(
        *,
        gateway: StorageGateway,
        repo: str,
        commit: str,
        limits: BackendLimits | None = None,
    ) -> ServiceApp:
        return build_service_app(
            gateway,
            repo=repo,
            commit=commit,
            limits=limits,
        )

    return _build


@pytest.fixture
def provisioned_service_app(
    provisioned_repo: ProvisionedGateway,
    service_app_factory: Callable[..., ServiceApp],
) -> ServiceApp:
    """Service app constructed from the provisioned_repo fixture."""
    return service_app_factory(
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )


@pytest.fixture
def architecture_service_app(
    architecture_gateway: StorageGateway,
    service_app_factory: Callable[..., ServiceApp],
) -> ServiceApp:
    """Service app constructed from the architecture_gateway fixture."""
    return service_app_factory(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )


__all__ = [
    "architecture_service_app",
    "provisioned_service_app",
    "service_app_factory",
]
