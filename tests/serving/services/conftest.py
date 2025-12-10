"""Shared fixtures for service delegate tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from codeintel.serving.backend import BackendLimits
from codeintel.storage.gateway import StorageGateway
from tests._helpers.analytics_samples import AnalyticsSamples, load_analytics_samples
from tests._helpers.serving_apps import ServiceApp, build_service_app

if TYPE_CHECKING:
    from tests._helpers import ProvisionedGateway


@pytest.fixture
def service_app_factory() -> Callable[..., ServiceApp]:
    """Build service apps bound to a gateway snapshot.

    Returns
    -------
    Callable[..., ServiceApp]
        Factory that produces configured ServiceApp instances.
    """

    def _build(
        *,
        gateway: StorageGateway,
        repo: str,
        commit: str,
        limits: BackendLimits | None = None,
    ) -> ServiceApp:
        return build_service_app(
            gateway,
            snapshot=(repo, commit),
            limits=limits,
        )

    return _build


@pytest.fixture
def provisioned_service_app(
    provisioned_repo: ProvisionedGateway,
    service_app_factory: Callable[..., ServiceApp],
) -> ServiceApp:
    """Service app constructed from the provisioned_repo fixture.

    Returns
    -------
    ServiceApp
        Configured service app for provisioned repo snapshots.
    """
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
    """Service app constructed from the architecture_gateway fixture.

    Returns
    -------
    ServiceApp
        Configured service app for architecture-focused tests.
    """
    return service_app_factory(
        gateway=architecture_gateway,
        repo="demo/repo",
        commit="deadbeef",
    )


@pytest.fixture
def analytics_samples(
    provisioned_service_app: ServiceApp,
) -> AnalyticsSamples:
    """Analytics identifiers extracted once per session.

    Returns
    -------
    AnalyticsSamples
        Sample identifiers from the provisioned service gateway.
    """
    return load_analytics_samples(provisioned_service_app.gateway)


@pytest.fixture
def architecture_samples(
    architecture_service_app: ServiceApp,
) -> AnalyticsSamples:
    """Analytics identifiers for architecture-focused service app.

    Returns
    -------
    AnalyticsSamples
        Sample identifiers from the architecture gateway.
    """
    return load_analytics_samples(architecture_service_app.gateway)


__all__ = [
    "architecture_service_app",
    "provisioned_service_app",
    "service_app_factory",
]
