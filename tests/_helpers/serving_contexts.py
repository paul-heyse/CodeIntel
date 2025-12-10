"""Provisioned service contexts with rebuild helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from codeintel.serving.backend import BackendLimits
from codeintel.serving.services.observability import ServiceObservability
from codeintel.storage.gateway import StorageGateway
from tests._helpers.serving_apps import (
    DEFAULT_LIMIT,
    MAX_ROWS,
    ServiceContext,
    build_service_context_from_components,
)

if TYPE_CHECKING:
    from tests.serving.mcp.conftest import McpBackendComponents


@dataclass(frozen=True)
class ProvisionedServiceContext:
    """Service context with helpers to rebuild under new limits or gateways."""

    context: ServiceContext
    _factory: Callable[[StorageGateway, BackendLimits], McpBackendComponents] | None

    @property
    def gateway(self) -> StorageGateway:
        return self.context.gateway

    @property
    def repo(self) -> str:
        return self.context.repo

    @property
    def commit(self) -> str:
        return self.context.commit

    @property
    def limits(self) -> BackendLimits:
        return self.context.limits

    @contextmanager
    def client(self) -> Iterator[TestClient]:
        """Provide a TestClient bound to the underlying app.

        Yields
        ------
        Iterator[TestClient]
            Client bound to the provisioned service application.
        """
        with TestClient(self.context.app) as client:
            yield client

    def with_limits(self, limits: BackendLimits) -> ProvisionedServiceContext:
        """Return a new context rebuilt with the provided limits.

        Returns
        -------
        ProvisionedServiceContext
            New context bound to the same gateway with updated limits.

        Raises
        ------
        ValueError
            If the context cannot be rebuilt without a factory.
        """
        if self._factory is None:
            message = "Cannot rebuild service context without a factory"
            raise ValueError(message)
        components = self._factory(self.gateway, limits)
        new_ctx = build_service_context_from_components(components)
        return ProvisionedServiceContext(context=new_ctx, _factory=self._factory)

    def with_gateway(self, gateway: StorageGateway) -> ProvisionedServiceContext:
        """Return a new context bound to a different gateway.

        Returns
        -------
        ProvisionedServiceContext
            New context bound to the provided gateway.

        Raises
        ------
        ValueError
            If the context cannot be rebuilt without a factory.
        """
        if self._factory is None:
            message = "Cannot rebuild service context without a factory"
            raise ValueError(message)
        components = self._factory(gateway, self.limits)
        new_ctx = build_service_context_from_components(components)
        return ProvisionedServiceContext(context=new_ctx, _factory=self._factory)


def build_provisioned_service_context(
    mcp_backend_factory: Callable[..., McpBackendComponents],
    *,
    gateway: StorageGateway,
    snapshot: tuple[str, str],
    limits: BackendLimits | None = None,
    observability: ServiceObservability | None = None,
) -> ProvisionedServiceContext:
    """Create a ProvisionedServiceContext from a gateway snapshot.

    Returns
    -------
    ProvisionedServiceContext
        Context backed by MCP components for the given snapshot.
    """
    effective_limits = limits or BackendLimits(
        default_limit=DEFAULT_LIMIT,
        max_rows_per_call=MAX_ROWS,
    )
    repo, commit = snapshot

    def _factory(
        current_gateway: StorageGateway,
        current_limits: BackendLimits,
    ) -> McpBackendComponents:
        return mcp_backend_factory(
            gateway=current_gateway,
            repo=repo,
            commit=commit,
            limits=current_limits,
            observability=observability,
        )

    components = _factory(gateway, effective_limits)
    context = build_service_context_from_components(components, observability=observability)
    return ProvisionedServiceContext(context=context, _factory=_factory)


def build_provisioned_service_context_from_components(
    components: McpBackendComponents,
) -> ProvisionedServiceContext:
    """Wrap prebuilt MCP components in a ProvisionedServiceContext.

    Returns
    -------
    ProvisionedServiceContext
        Context backed by the provided components.
    """
    context = build_service_context_from_components(components)
    return ProvisionedServiceContext(context=context, _factory=None)


__all__ = [
    "ProvisionedServiceContext",
    "build_provisioned_service_context",
    "build_provisioned_service_context_from_components",
]
